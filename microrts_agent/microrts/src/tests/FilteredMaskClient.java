package tests;

import java.util.*;
import rts.*;
import rts.units.*;

/**
 * Computes action masks with destination-aware filtering.
 *
 * The original getMasks(player) in JNIGridnetClient only checks whether
 * a destination cell is terrain-walkable and has no unit currently sitting
 * on it. It does NOT check whether another unit is already moving to that
 * cell (i.e., has a pending action targeting it). This class adds that
 * check by building a ResourceUsage from all pending UnitActionAssignments
 * and filtering move/produce actions via consistentWith().
 *
 * Does NOT modify the original JNIGridnetClient, JNIGridnetClientSelfPlay,
 * or any MicroRTS engine class. Relies on package-private field access
 * (same 'tests' package).
 */
public class FilteredMaskClient
{
    /**
     * Compute destination-aware masks for a single JNIGridnetClient (bot env).
     * Reads gs, pgs, utt, masks, maxAttackRadius from the client.
     *
     * @param client the bot environment client
     * @param player which player to compute masks for (0 or 1)
     * @return the filtered mask array [height][width][mask_dim]
     */
    public static int[][][] getMasksFiltered(JNIGridnetClient client,
        int player) throws Exception
    {
        int[][][] masks = client.masks;
        GameState gs = client.gs;
        PhysicalGameState pgs = client.pgs;
        UnitTypeTable utt = client.utt;
        int maxAttackRadius = client.maxAttackRadius;

        // Zero out masks (same as original getMasks)
        for (int y = 0; y < masks.length; y++)
        {
            for (int x = 0; x < masks[0].length; x++)
            {
                Arrays.fill(masks[y][x], 0);
            }
        }

        // Build ResourceUsage from pending actions of this player only.
        // Enemy pending actions should not block our moves — the engine
        // resolves inter-player destination conflicts at cycle time.
        ResourceUsage baseRU = new ResourceUsage();
        for (UnitActionAssignment uaa : gs.getUnitActions().values())
        {
            if (uaa.unit.getPlayer() == player)
            {
                ResourceUsage ru = uaa.action.resourceUsage(uaa.unit, pgs);
                baseRU.merge(ru);
            }
        }

        // Fill masks for idle units owned by 'player', filtering by ResourceUsage
        for (Unit u : pgs.getUnits())
        {
            if (u.getPlayer() == player && gs.getActionAssignment(u) == null)
            {
                masks[u.getY()][u.getX()][0] = 1;  // source unit bit
                fillFilteredMask(u, gs, utt, masks[u.getY()][u.getX()],
                                 maxAttackRadius, 1, baseRU);
            }
        }
        return masks;
    }

    /**
     * Compute destination-aware masks for a JNIGridnetClientSelfPlay (self-play env).
     * The selfplay client has a 4D masks array: masks[player][y][x][mask_dim].
     *
     * @param client the self-play environment client
     * @param player which player to compute masks for (0 or 1)
     * @return the filtered mask array [height][width][mask_dim]
     */
    public static int[][][] getMasksFiltered(JNIGridnetClientSelfPlay client,
        int player) throws Exception
    {
        int[][][] masks = client.masks[player];
        GameState gs = client.gs;
        PhysicalGameState pgs = client.pgs;
        UnitTypeTable utt = client.utt;
        int maxAttackRadius = client.maxAttackRadius;

        // Zero out masks
        for (int y = 0; y < masks.length; y++)
        {
            for (int x = 0; x < masks[0].length; x++)
            {
                Arrays.fill(masks[y][x], 0);
            }
        }

        // Build ResourceUsage from pending actions of this player only
        ResourceUsage baseRU = new ResourceUsage();
        for (UnitActionAssignment uaa : gs.getUnitActions().values())
        {
            if (uaa.unit.getPlayer() == player)
            {
                ResourceUsage ru = uaa.action.resourceUsage(uaa.unit, pgs);
                baseRU.merge(ru);
            }
        }

        // Fill masks for idle units owned by 'player'
        for (Unit u : pgs.getUnits())
        {
            if (u.getPlayer() == player && gs.getActionAssignment(u) == null)
            {
                masks[u.getY()][u.getX()][0] = 1;
                fillFilteredMask(u, gs, utt, masks[u.getY()][u.getX()],
                                 maxAttackRadius, 1, baseRU);
            }
        }
        return masks;
    }

    /**
     * Return a 2D grid [height][width] with 1 at each cell reserved by a
     * pending move/produce action. Used on the Python side to add an extra
     * observation channel so the neural network can "see" where units are
     * moving to or being produced.
     *
     * @param gs the current GameState
     * @return int[height][width] — 1 = reserved, 0 = free
     */
    public static int[][] getReservedPositions(GameState gs)
    {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int w = pgs.getWidth();
        int h = pgs.getHeight();
        int[][] grid = new int[h][w];

        for (UnitActionAssignment uaa : gs.getUnitActions().values())
        {
            ResourceUsage ru = uaa.action.resourceUsage(uaa.unit, pgs);
            for (Integer pos : ru.getPositionsUsed())
            {
                int py = pos / w;
                int px = pos % w;
                grid[py][px] = 1;
            }
        }
        return grid;
    }

    /**
     * Fill the mask array for a single unit, filtering out move/produce actions
     * whose destination is already reserved by a pending action.
     *
     * This mirrors UnitAction.getValidActionArray but adds a
     * ResourceUsage.consistentWith() check for TYPE_MOVE and TYPE_PRODUCE.
     *
     * Mask encoding (offset=1):
     *   [1..6]  action types (NONE=0, MOVE=1, HARVEST=2, RETURN=3, PRODUCE=4, ATTACK=5)
     *   [7..10] move direction (UP=0, RIGHT=1, DOWN=2, LEFT=3)
     *   [11..14] harvest direction
     *   [15..18] return direction
     *   [19..22] produce direction
     *   [23..22+N] produce unit type (N = numUnitTypes)
     *   [23+N..] attack location grid (maxAttackRadius x maxAttackRadius)
     */
    private static void fillFilteredMask(Unit u, GameState gs, UnitTypeTable utt,
        int[] mask, int maxAttackRange, int offset, ResourceUsage baseRU)
    {
        List<UnitAction> uas = u.getUnitActions(gs);
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int numUnitTypes = utt.getUnitTypes().size();
        int halfRange = maxAttackRange / 2;

        for (UnitAction ua : uas)
        {
            int aType = ua.getType();

            // Filter move and produce: skip if destination is already reserved
            if (aType == UnitAction.TYPE_MOVE || aType == UnitAction.TYPE_PRODUCE)
            {
                ResourceUsage uaRU = ua.resourceUsage(u, pgs);
                if (!uaRU.consistentWith(baseRU, gs)) { continue; }
            }

            // Encode into mask (same layout as UnitAction.getValidActionArray)
            mask[offset + aType] = 1;
            switch (aType)
            {
                case UnitAction.TYPE_NONE:
                    break;
                case UnitAction.TYPE_MOVE:
                    mask[offset + 6 + ua.getDirection()] = 1;
                    break;
                case UnitAction.TYPE_HARVEST:
                    mask[offset + 6 + 4 + ua.getDirection()] = 1;
                    break;
                case UnitAction.TYPE_RETURN:
                    mask[offset + 6 + 4 + 4 + ua.getDirection()] = 1;
                    break;
                case UnitAction.TYPE_PRODUCE:
                    mask[offset + 6 + 4 + 4 + 4 + ua.getDirection()] = 1;
                    mask[offset + 6 + 4 + 4 + 4 + 4 + ua.getUnitType().ID] = 1;
                    break;
                case UnitAction.TYPE_ATTACK_LOCATION:
                    int dx = ua.getLocationX() - u.getX();
                    int dy = ua.getLocationY() - u.getY();
                    mask[offset + 6 + 4 + 4 + 4 + 4 + numUnitTypes
                        + (halfRange + dy) * maxAttackRange
                        + (halfRange + dx)] = 1;
                    break;
            }
        }
    }
}
