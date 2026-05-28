/*
 * Bot-vs-bot bridge: 2 Java AIs play against each other (no RL agent).
 * Used by JNIGridnetVecClient for evaluation / tournament games.
 *
 * Unlike JNIGridnetClient, gameStep() takes NO actions from Python —
 * both AIs decide autonomously. Python only calls gameStep(player)
 * to advance the game and read rewards/dones.
 *
 * No observations are returned (obs = null in Response) since
 * there's no neural network to feed.
 */
package tests;

import ai.core.AI;
import ai.jni.Response;
import ai.reward.RewardFunctionInterface;
import gui.PhysicalGameStateJFrame;
import gui.PhysicalGameStatePanel;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.StringWriter;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import ai.wrapper.GameStateWrapper;
import rts.GameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.Trace;
import rts.TraceEntry;
import rts.units.UnitTypeTable;

public class JNIBotClient
{
    // --- Core state ---
    public AI ai1;                          // first bot (plays as `player`)
    public AI ai2;                          // second bot (plays as `1 - player`)
    PhysicalGameState pgs;
    public GameState gs;
    UnitTypeTable utt;
    boolean partialObs;
    public RewardFunctionInterface[] rfs;
    public String mapPath;
    String micrortsPath;
    boolean gameover = false;
    double[] rewards;
    boolean[] dones;
    Response response;
    PlayerAction pa1;
    PlayerAction pa2;

    // --- Rendering ---
    boolean layerJSON = true;
    public int renderTheme = PhysicalGameStatePanel.COLORSCHEME_WHITE;
    PhysicalGameStateJFrame w;

    // --- Trace recording ---
    public List<TraceEntry> traceEntries = null;
    public boolean collectTrace = false;

    // --- Bot timing ---
    public long ai1TotalTimeNs = 0L;
    public long ai2TotalTimeNs = 0L;
    public int ai1TimeoutCount = 0;
    public int ai2TimeoutCount = 0;
    public long timeBudgetMs = 100L;

    /**
     * @param rfs           reward functions
     * @param micrortsPath  root directory
     * @param mapPath       map XML file
     * @param ai1           first bot
     * @param ai2           second bot
     * @param utt           unit type table
     * @param partialObs    enable fog of war
     */
    public JNIBotClient(RewardFunctionInterface[] rfs, String micrortsPath,
            String mapPath, AI ai1, AI ai2, UnitTypeTable utt, boolean partialObs) throws Exception
    {
        this.micrortsPath = micrortsPath;
        this.mapPath = mapPath;
        this.rfs = rfs;
        this.utt = utt;
        this.partialObs = partialObs;
        this.ai1 = ai1;
        this.ai2 = ai2;
        if (ai1 == null || ai2 == null) { throw new Exception("no ai1 or ai2 was chosen"); }
        if (micrortsPath.length() != 0) { this.mapPath = Paths.get(micrortsPath, this.mapPath).toString(); }
        pgs = PhysicalGameState.load(this.mapPath, utt);
        rewards = new double[rfs.length];
        dones = new boolean[rfs.length];
        response = new Response(null, null, null, null);
    }

    // ------------------------------------------------------------------
    // Core methods
    // ------------------------------------------------------------------

    /**
     * Both bots pick actions autonomously, advance 1 tick, compute rewards.
     * No obs returned (obs = null) since there's no neural network.
     *
     * @param player  which player's perspective for reward computation
     * @return Response(null, rewards, dones, "{}")
     */
    public Response gameStep(int player) throws Exception
    {
        // Both bots compute their actions (timed for timeout tracking)
        long t0 = System.nanoTime();
        pa1 = ai1.getAction(player, gs);
        long t1 = System.nanoTime();
        pa2 = ai2.getAction(1 - player, gs);
        long t2 = System.nanoTime();

        ai1TotalTimeNs += t1 - t0;
        ai2TotalTimeNs += t2 - t1;
        if ((t1 - t0) / 1_000_000L > timeBudgetMs) ai1TimeoutCount++;
        if ((t2 - t1) / 1_000_000L > timeBudgetMs) ai2TimeoutCount++;

        // Issue actions and advance
        gs.issueSafe(pa1);
        gs.issueSafe(pa2);

        // Cache vector actions BEFORE cycle() clears action assignments
        lastVectorAction1 = GameStateWrapper.toVectorAction(gs, pa1);
        lastVectorAction2 = GameStateWrapper.toVectorAction(gs, pa2);

        TraceEntry te = new TraceEntry(gs.getPhysicalGameState().clone(), gs.getTime());
        te.addPlayerAction(pa1.clone());
        te.addPlayerAction(pa2.clone());
        if (collectTrace && traceEntries != null) { traceEntries.add(te); }
        gameover = gs.cycle();
        if (gameover)
        {
            ai1.gameOver(gs.winner());
            ai2.gameOver(gs.winner());
        }

        // Compute rewards
        for (int i = 0; i < rewards.length; i++)
        {
            rfs[i].computeReward(player, 1 - player, te, gs);
            dones[i] = rfs[i].isDone();
            rewards[i] = rfs[i].getReward();
        }
        response.set(null, rewards, dones, "{}");
        return response;
    }

    /** Return UTT as JSON string. */
    public String sendUTT() throws Exception
    {
        StringWriter sw = new StringWriter();
        utt.toJSON(sw);
        return sw.toString();
    }

    /**
     * Reset: clone + reset both bots, reload map, zero rewards.
     *
     * @param player  which player's perspective
     * @return Response with zeroed rewards/dones (obs = null)
     */
    public Response reset(int player) throws Exception
    {
        ai1 = ai1.clone();
        ai1.reset();
        ai2 = ai2.clone();
        ai2.reset();
        pgs = PhysicalGameState.load(mapPath, utt);
        gs = new GameState(pgs, utt);
        for (int i = 0; i < rewards.length; i++)
        {
            rewards[i] = 0.0;
            dones[i] = false;
        }
        ai1TotalTimeNs = 0L;
        ai2TotalTimeNs = 0L;
        ai1TimeoutCount = 0;
        ai2TimeoutCount = 0;
        if (collectTrace) { traceEntries = new ArrayList<>(); }
        response.set(null, rewards, dones, "{}");
        return response;
    }

    // ------------------------------------------------------------------
    // Observation & action extraction (for behavior cloning)
    // ------------------------------------------------------------------

    /**
     * Get the raw observation from the current game state.
     * Returns int[F][H][W] — same format as JNIAI.getObservation().
     */
    public int[][][] getObservation(int player) {
        return gs.getVectorObservation(player);
    }

    /**
     * Get the last action played by the given player as a vector.
     * Returns int[numUnits][8] — same format as Python's action grid
     * (cellIdx, actionType, moveDir, harvestDir, returnDir, produceDir, produceType, attackTarget).
     * Returns empty array if no action was played yet.
     *
     * Uses cached pre-cycle vectors computed in gameStep() to avoid
     * NullPointerException from getActionAssignment() after gs.cycle().
     */
    public int[][] getLastAction(int player) {
        return (player == 0) ? lastVectorAction1 : lastVectorAction2;
    }

    private int[][] lastVectorAction1 = new int[0][];
    private int[][] lastVectorAction2 = new int[0][];

    // ------------------------------------------------------------------
    // Rendering
    // ------------------------------------------------------------------

    /** Render the game state to pixels or GUI window. */
    public byte[] render(boolean returnPixels) throws Exception
    {
        if (w == null) { w = PhysicalGameStatePanel.newVisualizer(gs, 640, 640, false, null, renderTheme); }
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
        if (w != null) { w.dispose(); }
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
