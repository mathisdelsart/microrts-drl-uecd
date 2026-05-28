/*
 * The RL agent's representation on the Java side.
 * Implements JNIInterface so JNIGridnetClient can call it uniformly.
 *
 * This is NOT a real AI — it doesn't decide anything. It just:
 *   1. Converts Python's int[][] action grid into a Java PlayerAction
 *   2. Extracts raw observations from the GameState
 *
 * Called by JNIGridnetClient.gameStep() every tick.
 */
package ai.jni;

import ai.core.AI;
import ai.core.AIWithComputationBudget;
import ai.core.ParameterSpecification;
import java.util.List;
import rts.GameState;
import rts.PlayerAction;
import rts.units.UnitTypeTable;

public class JNIAI extends AIWithComputationBudget implements JNIInterface
{
    UnitTypeTable utt;
    int maxAttackRadius;  // diameter = attackRange * 2 + 1 (for attack target grid)

    /**
     * @param timeBudget  time budget in ms — unused, passed to parent
     * @param iterBudget  iterations budget — unused, passed to parent
     * @param utt         the UTT (defines unit types, attack ranges, etc.)
     */
    public JNIAI(int timeBudget, int iterBudget, UnitTypeTable utt)
    {
        super(timeBudget, iterBudget);
        this.utt = utt;
        this.maxAttackRadius = utt.getMaxAttackRange() * 2 + 1;
    }

    /**
     * Convert Python's action grid into a Java PlayerAction.
     *
     * @param player     player id (0 or 1)
     * @param gs         current game state
     * @param actions    int[numUnits][8] — each row:
     *                   [cellIdx, actionType, moveDir, harvestDir,
     *                    returnDir, produceDir, produceType, attackTarget]
     * @return PlayerAction ready to be issued via gs.issueSafe()
     *
     * fromVectorAction() maps each row to a UnitAction for the unit at that cell.
     * fillWithNones() assigns TYPE_NONE to any idle unit not in actions[],
     * so every unit has an action (required by the engine).
     */
    @Override
    public PlayerAction getAction(int player, GameState gs, int[][] actions) throws Exception
    {
        PlayerAction pa = PlayerAction.fromVectorAction(actions, gs, this.utt, player, this.maxAttackRadius);
        pa.fillWithNones(gs, player, 1);
        return pa;
    }

    /**
     * Extract raw observations from the game state.
     * Returns int[13][H][W] — see GameStateWrapper.getVectorObservation() for
     * the full feature map (HP, resources, owner, unitType, actions, ETA, terrain).
     *
     * This is what Python receives as "raw obs" before encoding into 29ch or 73ch.
     */
    @Override
    public int[][][] getObservation(int player, GameState gs) throws Exception
    {
        return gs.getVectorObservation(player);
    }

    @Override
    public void reset() { }

    // --- Stubs required by AI / JNIInterface but not used for RL ---

    /** Not used — rewards are computed by RewardFunctionInterface[] in JNIGridnetClient. */
    @Override
    public double computeReward(int maxplayer, int minplayer, GameState gs) { return 0.0; }

    /** Not used — RL agent receives actions from Python, not from this method. */
    @Override
    public PlayerAction getAction(int player, GameState gs) { return null; }

    @Override
    public String computeInfo(int player, GameState gs) { return null; }

    @Override
    public AI clone() { return null; }

    @Override
    public List<ParameterSpecification> getParameters() { return null; }
}
