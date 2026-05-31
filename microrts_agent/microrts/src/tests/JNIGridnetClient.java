/*
 * Single-env bridge: 1 RL agent (player) vs 1 Java bot (ai2).
 * Used by JNIGridnetVecClient for bot environments.
 *
 * Lifecycle:  reset(player) -> [gameStep(actions, player) -> getMasks(player)]* -> close()
 *
 * Flow per gameStep:
 *   1. Build partial-obs views if needed
 *   2. JNIAI converts Python's int[][] actions -> PlayerAction
 *   3. Bot computes its own action (timed for timeout tracking)
 *   4. Both actions issued to GameState, game advances 1 tick
 *   5. All RewardFunctions compute their signals
 *   6. Response(obs, rewards, dones, info) returned to Python
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

public class JNIGridnetClient
{
    // --- Core state ---
    public RewardFunctionInterface[] rfs;   // reward functions (WinDrawLoss, Attack, etc.)
    String micrortsPath;                    // root path to microrts/ directory
    public String mapPath;                  // resolved map file path
    public AI ai2;                          // the opponent bot (Java AI)
    UnitTypeTable utt;                      // unit type table (shared)
    public boolean partialObs = false;      // fog of war enabled?
    public PhysicalGameState pgs;           // physical game state (map + units)
    public GameState gs;                    // full game state (pgs + actions + time)
    public GameState player1gs;             // RL agent's view (partial if partialObs)
    public GameState player2gs;             // bot's view (partial if partialObs)
    boolean gameover = false;
    public int maxAttackRadius;             // attack grid diameter = range * 2 + 1

    // --- Bridge objects ---
    public JNIInterface ai1;                // the RL agent proxy (JNIAI instance)
    int[][][] masks;                        // action mask [H][W][mask_dim]
    double[] rewards;                       // reward per reward function
    boolean[] dones;                        // done per reward function
    Response response;                      // reusable response container
    PlayerAction pa1;                       // RL agent's last action
    PlayerAction pa2;                       // bot's last action

    // --- Rendering ---
    boolean layerJSON = true;
    public int renderTheme = PhysicalGameStatePanel.COLORSCHEME_WHITE;
    PhysicalGameStateJFrame w;

    // --- Trace recording ---
    public List<TraceEntry> traceEntries = null;
    public boolean collectTrace = false;

    // --- Bot timing ---
    public long ai2TotalTimeNs = 0L;  // cumulative bot thinking time
    public int ai2TimeoutCount = 0;   // number of times bot exceeded budget
    public long timeBudgetMs = 100L;  // timeout threshold (ms)

    /**
     * @param rfs           reward functions to evaluate each step
     * @param micrortsPath  root directory (prepended to mapPath if non-empty)
     * @param mapPath       map XML file (relative to micrortsPath)
     * @param ai2           opponent bot AI
     * @param utt           unit type table
     * @param partialObs    enable fog of war
     */
    public JNIGridnetClient(RewardFunctionInterface[] rfs, String micrortsPath,
            String mapPath, AI ai2, UnitTypeTable utt, boolean partialObs) throws Exception
    {
        this.micrortsPath = micrortsPath;
        this.mapPath = mapPath;
        this.rfs = rfs;
        this.utt = utt;
        this.partialObs = partialObs;
        this.maxAttackRadius = utt.getMaxAttackRange() * 2 + 1;
        this.ai1 = new JNIAI(100, 0, utt);
        this.ai2 = ai2;
        if (this.ai2 == null) { throw new Exception("no ai2 was chosen"); }
        if (micrortsPath.length() != 0)
        {
            this.mapPath = Paths.get(micrortsPath, this.mapPath).toString();
        }
        this.pgs = PhysicalGameState.load(this.mapPath, utt);
        this.masks = new int[pgs.getHeight()][pgs.getWidth()]
                [23 + utt.getUnitTypes().size() + maxAttackRadius * maxAttackRadius];
        this.rewards = new double[rfs.length];
        this.dones = new boolean[rfs.length];
        this.response = new Response(null, null, null, null);
    }

    // ------------------------------------------------------------------
    // Core methods
    // ------------------------------------------------------------------

    /**
     * Advance the game by one tick.
     *
     * @param actions  int[numUnits][8] action array from Python
     * @param player   which player the RL agent controls (0 or 1)
     * @return Response containing new obs, rewards, dones, info
     */
    public Response gameStep(int[][] actions, int player) throws Exception
    {
        // 1. Build per-player views (partial obs = fog of war)
        if (this.partialObs)
        {
            this.player1gs = new PartiallyObservableGameState(gs, player);
            this.player2gs = new PartiallyObservableGameState(gs, 1 - player);
        }
        else
        {
            this.player1gs = gs;
            this.player2gs = gs;
        }

        // 2. RL agent: convert int[][] -> PlayerAction via JNIAI
        this.pa1 = ai1.getAction(player, player1gs, actions);

        // 3. Bot: compute its own action (timed for timeout tracking)
        long t0 = System.nanoTime();
        try { this.pa2 = ai2.getAction(1 - player, player2gs); }
        catch (Exception e)
        {
            System.out.println("AI crash on map: " + mapPath);
            e.printStackTrace(System.out);
            throw e;
        }
        long t1 = System.nanoTime();
        ai2TotalTimeNs += t1 - t0;
        if ((t1 - t0) / 1_000_000L > timeBudgetMs) { ai2TimeoutCount++; }

        // 4. Issue both actions and advance the game by one cycle
        gs.issueSafe(pa1);
        gs.issueSafe(pa2);
        TraceEntry te = new TraceEntry(gs.getPhysicalGameState().clone(), gs.getTime());
        te.addPlayerAction(pa1.clone());
        te.addPlayerAction(pa2.clone());
        if (collectTrace && traceEntries != null) { traceEntries.add(te); }
        gameover = gs.cycle();
        if (gameover) { ai2.gameOver(gs.winner()); }

        // 5. Compute all reward signals
        for (int i = 0; i < rewards.length; i++)
        {
            rfs[i].computeReward(player, 1 - player, te, gs);
            dones[i] = rfs[i].isDone();
            rewards[i] = rfs[i].getReward();
        }

        // 6. Pack and return
        response.set(
            ai1.getObservation(player, player1gs),
            rewards, dones,
            ai1.computeInfo(player, player2gs)
        );
        return response;
    }

    /**
     * Valid action mask for the given player.
     *
     * @param player  which player (0 or 1)
     * @return int[H][W][mask_dim]: masks[y][x][0] = 1 means controllable unit at (y,x),
     *         remaining slots encode which action types/directions/targets are valid.
     */
    public int[][][] getMasks(int player) throws Exception
    {
        // Zero out all masks
        for (int y = 0; y < masks.length; y++)
        {
            for (int x = 0; x < masks[0].length; x++)
            {
                Arrays.fill(masks[y][x], 0);
            }
        }
        // Fill masks for each idle unit owned by player
        for (Unit u : pgs.getUnits())
        {
            if (u.getPlayer() == player && gs.getActionAssignment(u) == null)
            {
                masks[u.getY()][u.getX()][0] = 1;
                UnitAction.getValidActionArray(u, gs, utt,
                    masks[u.getY()][u.getX()], maxAttackRadius, 1);
            }
        }
        return masks;
    }

    /** Return UTT as JSON string (sent to Python once at init). */
    public String sendUTT() throws Exception
    {
        StringWriter sw = new StringWriter();
        utt.toJSON(sw);
        return sw.toString();
    }

    /**
     * Reset the environment: reload map, reset both AIs, return initial obs.
     *
     * @param player  which player the RL agent controls
     * @return Response with initial obs and zeroed rewards/dones
     */
    public Response reset(int player) throws Exception
    {
        ai1.reset();
        ai2 = ai2.clone();
        ai2.reset();
        pgs = PhysicalGameState.load(mapPath, utt);
        masks = new int[pgs.getHeight()][pgs.getWidth()]
                [23 + utt.getUnitTypes().size() + maxAttackRadius * maxAttackRadius];
        gs = new GameState(pgs, utt);
        player1gs = partialObs ? new PartiallyObservableGameState(gs, player) : gs;
        for (int i = 0; i < rewards.length; i++)
        {
            rewards[i] = 0.0;
            dones[i] = false;
        }
        ai2TotalTimeNs = 0L;
        ai2TimeoutCount = 0;
        response.set(ai1.getObservation(player, player1gs), rewards, dones, "{}");
        return response;
    }

    // ------------------------------------------------------------------
    // Rendering
    // ------------------------------------------------------------------

    /**
     * Render the game state.
     *
     * @param returnPixels  true = return RGB byte array, false = update GUI window
     * @return byte[] of pixel data (TYPE_3BYTE_BGR), or null if GUI mode
     */
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

    /** Dispose the GUI window. Does NOT shut down the JVM. */
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
