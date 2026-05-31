/*
 * Self-play bridge: 2 RL agents on the same map (P0 vs P1).
 * Used by JNIGridnetVecClient for selfplay environments.
 *
 * Unlike JNIGridnetClient, there is no bot: both players are controlled
 * by Python. gameStep() takes TWO action arrays (one per player).
 *
 * Each player gets its own JNIAI instance, masks, rewards, dones, and Response.
 * Python reads results via getResponse(player) after gameStep().
 */
package tests;

import ai.core.AI;
import ai.jni.JNIAI;
import ai.jni.JNIInterface;
import ai.jni.Response;
import ai.reward.RewardFunctionInterface;
import gui.PhysicalGameStateJFrame;
import gui.PhysicalGameStatePanel;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.StringWriter;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import rts.GameState;
import rts.PartiallyObservableGameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.Trace;
import rts.TraceEntry;
import rts.UnitAction;
import rts.units.Unit;
import rts.units.UnitTypeTable;

public class JNIGridnetClientSelfPlay
{
    // --- Core state ---
    public RewardFunctionInterface[] rfs;
    String micrortsPath;
    public String mapPath;
    public AI ai2;                                  // unused in self-play, kept for API compat
    UnitTypeTable utt;
    boolean partialObs = false;
    public PhysicalGameState pgs;
    public GameState gs;
    public GameState[] playergs = new GameState[2]; // per-player views (partial if fog of war)
    boolean gameover = false;
    public int maxAttackRadius;
    public int numPlayers = 2;

    // --- Per-player bridge objects ---
    public JNIInterface[] ais = new JNIInterface[2];    // one JNIAI per player
    int[][][][] masks = new int[2][][][];               // [player][H][W][mask_dim]
    double[][] rewards = new double[2][];               // [player][N_reward_functions]
    boolean[][] dones = new boolean[2][];               // [player][N_reward_functions]
    Response[] response = new Response[2];
    PlayerAction[] pas = new PlayerAction[2];

    // --- Rendering ---
    boolean layerJSON = true;
    public int renderTheme = PhysicalGameStatePanel.COLORSCHEME_WHITE;
    PhysicalGameStateJFrame w;

    // --- Trace recording ---
    public List<TraceEntry> traceEntries = null;
    public boolean collectTrace = false;

    /**
     * @param rfs           reward functions (shared between both players)
     * @param micrortsPath  root directory
     * @param mapPath       map XML file
     * @param utt           unit type table
     * @param partialObs    enable fog of war
     */
    public JNIGridnetClientSelfPlay(RewardFunctionInterface[] rfs, String micrortsPath,
            String mapPath, UnitTypeTable utt, boolean partialObs) throws Exception
    {
        this.micrortsPath = micrortsPath;
        this.mapPath = mapPath;
        this.rfs = rfs;
        this.utt = utt;
        this.partialObs = partialObs;
        this.maxAttackRadius = utt.getMaxAttackRange() * 2 + 1;
        if (micrortsPath.length() != 0) { this.mapPath = Paths.get(micrortsPath, this.mapPath).toString(); }
        pgs = PhysicalGameState.load(this.mapPath, utt);
        for (int p = 0; p < numPlayers; p++)
        {
            ais[p] = new JNIAI(100, 0, utt);
            masks[p] = new int[pgs.getHeight()][pgs.getWidth()]
                    [23 + utt.getUnitTypes().size() + maxAttackRadius * maxAttackRadius];
            rewards[p] = new double[rfs.length];
            dones[p] = new boolean[rfs.length];
            response[p] = new Response(null, null, null, null);
        }
    }

    // ------------------------------------------------------------------
    // Core methods
    // ------------------------------------------------------------------

    /**
     * Advance 1 tick with both players' actions.
     *
     * Both actions are issued before gs.cycle(), so the game resolves
     * them simultaneously (same as real MicroRTS).
     *
     * @param actionsP0  int[][] actions for player 0
     * @param actionsP1  int[][] actions for player 1
     */
    public void gameStep(int[][] actionsP0, int[][] actionsP1) throws Exception
    {
        TraceEntry te = new TraceEntry(gs.getPhysicalGameState().clone(), gs.getTime());
        for (int p = 0; p < numPlayers; p++)
        {
            playergs[p] = gs;
            if (partialObs) { playergs[p] = new PartiallyObservableGameState(gs, p); }
            pas[p] = (p == 0)
                ? ais[p].getAction(p, playergs[0], actionsP0)
                : ais[p].getAction(p, playergs[1], actionsP1);
            gs.issueSafe(pas[p]);
            te.addPlayerAction(pas[p].clone());
        }

        if (collectTrace && traceEntries != null) { traceEntries.add(te); }
        gameover = gs.cycle();

        // Compute rewards for both players (each sees itself as maxplayer)
        for (int p = 0; p < numPlayers; p++)
        {
            for (int i = 0; i < rfs.length; i++)
            {
                rfs[i].computeReward(p, 1 - p, te, gs);
                rewards[p][i] = rfs[i].getReward();
                dones[p][i] = rfs[i].isDone();
            }
            response[p].set(
                ais[p].getObservation(p, playergs[p]),
                rewards[p], dones[p], "{}"
            );
        }
    }

    /**
     * Valid action mask for the given player.
     * Same logic as JNIGridnetClient.getMasks().
     */
    public int[][][] getMasks(int player) throws Exception
    {
        for (int y = 0; y < masks[0].length; y++)
        {
            for (int x = 0; x < masks[0][0].length; x++)
            {
                Arrays.fill(masks[player][y][x], 0);
            }
        }
        for (Unit u : pgs.getUnits())
        {
            if (u.getPlayer() == player && gs.getActionAssignment(u) == null)
            {
                masks[player][u.getY()][u.getX()][0] = 1;
                UnitAction.getValidActionArray(u, gs, utt,
                    masks[player][u.getY()][u.getX()], maxAttackRadius, 1);
            }
        }
        return masks[player];
    }

    /** Return UTT as JSON string. */
    public String sendUTT() throws Exception
    {
        StringWriter sw = new StringWriter();
        utt.toJSON(sw);
        return sw.toString();
    }

    /** Reset both players: reload map, zero rewards, return initial obs. */
    public void reset() throws Exception
    {
        pgs = PhysicalGameState.load(mapPath, utt);
        for (int p = 0; p < numPlayers; p++)
        {
            masks[p] = new int[pgs.getHeight()][pgs.getWidth()]
                    [23 + utt.getUnitTypes().size() + maxAttackRadius * maxAttackRadius];
        }
        gs = new GameState(pgs, utt);
        for (int p = 0; p < numPlayers; p++)
        {
            playergs[p] = gs;
            if (partialObs) { playergs[p] = new PartiallyObservableGameState(gs, p); }
            ais[p].reset();
            for (int i = 0; i < rfs.length; i++)
            {
                rewards[p][i] = 0.0;
                dones[p][i] = false;
            }
            response[p].set(
                ais[p].getObservation(p, playergs[p]),
                rewards[p], dones[p], "{}"
            );
        }
    }

    /** Get the Response for a specific player (call after gameStep or reset). */
    public Response getResponse(int player)
    {
        return response[player];
    }

    // ------------------------------------------------------------------
    // Rendering
    // ------------------------------------------------------------------

    /** Render the game state to pixels or GUI window. */
    public byte[] render(boolean returnPixels) throws Exception
    {
        if (w == null) { w = PhysicalGameStatePanel.newVisualizer(gs, 640, 640, partialObs, null, renderTheme); }
        w.setStateCloning(gs);
        if (!returnPixels)
        {
            w.repaint();
            return null;
        }
        BufferedImage img = new BufferedImage(w.getWidth(), w.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
        w.paint(img.getGraphics());
        return ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
    }

    /** Dispose the GUI window. */
    public void close() throws Exception
    {
        if (w != null) w.dispose();
    }

    // ------------------------------------------------------------------
    // Trace recording (for game replays)
    // ------------------------------------------------------------------

    /** Start recording game actions for replay. */
    public void startTrace()
    {
        traceEntries = new ArrayList<>();
        collectTrace = true;
    }

    /** Stop recording and discard the trace buffer. */
    public void stopTrace()
    {
        collectTrace = false;
        traceEntries = null;
    }

    /** Save the recorded trace to a zip file. */
    public void saveTraceToZip(String path)
    {
        if (traceEntries == null || traceEntries.isEmpty()) return;
        Trace trace = new Trace(utt);
        for (TraceEntry te : traceEntries) trace.addEntry(te);
        trace.toZip(path);
    }
}
