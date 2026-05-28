/*
 * Vectorized env manager: orchestrates N parallel game environments.
 * This is the top-level class called from Python (via JPype).
 *
 * Manages three types of sub-clients:
 *   - JNIGridnetClientSelfPlay[]  → self-play envs (RL agent vs RL agent)
 *   - JNIGridnetClient[]          → bot envs (RL agent vs Java bot)
 *   - JNIBotClient[]              → bot-vs-bot envs (evaluation only)
 *
 * Environment layout (selfplay envs first, then bot envs):
 *   env indices: [0..2*nSelfplay-1] = selfplay, [2*nSelfplay..N-1] = bot envs
 *
 * Auto-reset: when an env is done or exceeds maxSteps, it is automatically
 * reset. The terminal reward/done is preserved in the response so Python
 * sees the final reward AND the fresh obs from the new episode in the same step.
 */
package tests;

import ai.PassiveAI;
import ai.core.AI;
import ai.jni.Response;
import ai.jni.Responses;
import ai.reward.RewardFunctionInterface;
import rts.units.UnitTypeTable;

public class JNIGridnetVecClient
{
    // --- Sub-clients ---
    public JNIGridnetClient[] clients;                    // bot envs (agent vs bot)
    public JNIGridnetClientSelfPlay[] selfPlayClients;    // selfplay envs (agent vs agent)
    public JNIBotClient[] botClients;                     // bot-vs-bot (evaluation only)

    // --- Shared config ---
    public int maxSteps;
    public int[] envSteps;                                // step counter per env
    public RewardFunctionInterface[] rfs;
    public UnitTypeTable utt;
    boolean partialObs = false;
    public String[] mapPaths;

    // --- Batched output arrays ---
    int[][][][] masks;                                    // [N_envs][H][W][mask_dim]
    int[][][][] observation;                              // [N_envs][13][H][W]
    double[][] reward;                                    // [N_envs][N_reward_functions]
    boolean[][] done;                                     // [N_envs][N_reward_functions]
    Response[] rs;                                        // per-env response objects
    Responses responses;                                  // batched response container

    // --- Auto-reset temp buffers ---
    double[] terminalReward1;                             // saves terminal reward before reset
    boolean[] terminalDone1;                              // saves terminal done before reset
    double[] terminalReward2;                             // selfplay: terminal reward for player 2
    boolean[] terminalDone2;                              // selfplay: terminal done for player 2

    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /**
     * Constructor for RL training: selfplay + bot environments.
     *
     * Creates numSelfplayEnvs/2 selfplay games (each produces 2 env slots:
     * one for P0, one for P1) plus numBotEnvs bot games (agent vs Java AI).
     *
     * A dummy env is created to probe observation dimensions, then discarded.
     *
     * @param numSelfplayEnvs  number of self-play env slots (must be even)
     * @param numBotEnvs       number of bot envs (agent vs Java AI)
     * @param maxSteps         max steps before forced reset
     * @param rfs              reward functions (shared across all envs)
     * @param micrortsPath     path to microrts/ directory
     * @param mapPaths         map file per env (relative to micrortsPath)
     * @param ai2s             opponent bot per bot-env
     * @param utt              unit type table
     * @param partialObs       enable fog of war
     */
    public JNIGridnetVecClient(int numSelfplayEnvs, int numBotEnvs, int maxSteps,
            RewardFunctionInterface[] rfs, String micrortsPath, String[] mapPaths,
            AI[] ai2s, UnitTypeTable utt, boolean partialObs) throws Exception
    {
        this.maxSteps = maxSteps;
        this.utt = utt;
        this.rfs = rfs;
        this.partialObs = partialObs;
        this.mapPaths = mapPaths;
        int totalEnvs = numSelfplayEnvs + numBotEnvs;
        this.envSteps = new int[totalEnvs];

        // Create selfplay clients (each manages 1 game with 2 players)
        this.selfPlayClients = new JNIGridnetClientSelfPlay[numSelfplayEnvs / 2];
        for (int i = 0; i < selfPlayClients.length; i++)
        {
            selfPlayClients[i] = new JNIGridnetClientSelfPlay(
                rfs, micrortsPath, mapPaths[i * 2], utt, partialObs);
        }

        // Create bot clients
        this.clients = new JNIGridnetClient[numBotEnvs];
        for (int i = 0; i < clients.length; i++)
        {
            clients[i] = new JNIGridnetClient(
                rfs, micrortsPath, mapPaths[numSelfplayEnvs + i],
                ai2s[i], utt, partialObs);
        }

        // Probe obs shape from a dummy env (closed after)
        JNIGridnetClient probeClient = new JNIGridnetClient(
            rfs, micrortsPath, mapPaths[0], new PassiveAI(utt), utt, partialObs);
        Response probe = probeClient.reset(0);
        probeClient.close();
        int obsF = probe.observation.length;
        int obsH = probe.observation[0].length;
        int obsW = probe.observation[0][0].length;

        // Allocate shared output arrays
        this.masks = new int[totalEnvs][][][];
        this.observation = new int[totalEnvs][obsF][obsH][obsW];
        this.reward = new double[totalEnvs][rfs.length];
        this.done = new boolean[totalEnvs][rfs.length];
        this.terminalReward1 = new double[rfs.length];
        this.terminalDone1 = new boolean[rfs.length];
        this.terminalReward2 = new double[rfs.length];
        this.terminalDone2 = new boolean[rfs.length];
        this.responses = new Responses(null, null, null);
        this.rs = new Response[totalEnvs];
    }

    /**
     * Constructor for bot-vs-bot evaluation (no RL agent).
     *
     * No observations are returned (obs = null) since both sides are
     * Java AIs that decide autonomously.
     *
     * @param maxSteps         max steps before forced reset
     * @param rfs              reward functions
     * @param micrortsPath     path to microrts/ directory
     * @param mapPaths         map file per env
     * @param ai1s             first bot per env
     * @param ai2s             second bot per env
     * @param utt              unit type table
     * @param partialObs       enable fog of war
     */
    public JNIGridnetVecClient(int maxSteps, RewardFunctionInterface[] rfs,
            String micrortsPath, String[] mapPaths, AI[] ai1s, AI[] ai2s,
            UnitTypeTable utt, boolean partialObs) throws Exception
    {
        this.maxSteps = maxSteps;
        this.utt = utt;
        this.rfs = rfs;
        this.partialObs = partialObs;
        this.mapPaths = mapPaths;

        this.botClients = new JNIBotClient[ai2s.length];
        for (int i = 0; i < botClients.length; i++)
        {
            botClients[i] = new JNIBotClient(
                rfs, micrortsPath, mapPaths[i], ai1s[i], ai2s[i], utt, partialObs);
        }

        this.responses = new Responses(null, null, null);
        this.rs = new Response[ai2s.length];
        this.reward = new double[ai2s.length][rfs.length];
        this.done = new boolean[ai2s.length][rfs.length];
        this.envSteps = new int[ai2s.length];
        this.terminalReward1 = new double[rfs.length];
        this.terminalDone1 = new boolean[rfs.length];
    }

    // ------------------------------------------------------------------
    // Core methods
    // ------------------------------------------------------------------

    /**
     * Reset all environments and return initial observations/rewards/dones.
     *
     * In bot-vs-bot mode, obs is null (no neural network).
     * In training mode, resets selfplay envs first, then bot envs.
     *
     * @param players  player assignment per env (0 or 1)
     * @return Responses with initial obs and zeroed rewards/dones
     */
    public Responses reset(int[] players) throws Exception
    {
        // Bot-vs-bot mode: no observations
        if (botClients != null)
        {
            for (int i = 0; i < botClients.length; i++)
            {
                rs[i] = botClients[i].reset(players[i]);
            }
            for (int i = 0; i < rs.length; i++)
            {
                reward[i] = rs[i].reward;
                done[i] = rs[i].done;
            }
            responses.set(null, reward, done);
            return responses;
        }

        // Selfplay envs: reset each game, read both players' responses
        for (int i = 0; i < selfPlayClients.length; i++)
        {
            selfPlayClients[i].reset();
            rs[i * 2]     = selfPlayClients[i].getResponse(0);
            rs[i * 2 + 1] = selfPlayClients[i].getResponse(1);
        }

        // Bot envs: reset each game
        for (int i = selfPlayClients.length * 2; i < players.length; i++)
        {
            rs[i] = clients[i - selfPlayClients.length * 2].reset(players[i]);
        }

        // Collect into shared arrays
        for (int i = 0; i < rs.length; i++)
        {
            observation[i] = rs[i].observation;
            reward[i] = rs[i].reward;
            done[i] = rs[i].done;
        }
        responses.set(observation, reward, done);
        return responses;
    }

    /**
     * Step all environments, with auto-reset on done or maxSteps exceeded.
     *
     * Auto-reset trick: when an env finishes, we save its terminal
     * reward/done, reset it (getting fresh obs), then overwrite the
     * response with the saved terminal values. Python sees the final
     * reward AND gets fresh obs for the next episode in the same step.
     *
     * @param actions  int[N_envs][H*W][7] action arrays from Python
     * @param players  player assignment per env (0 or 1)
     * @return Responses with obs/rewards/dones for all envs
     */
    public Responses gameStep(int[][][] actions, int[] players) throws Exception
    {
        // ---- Bot-vs-bot mode ----
        if (botClients != null)
        {
            for (int i = 0; i < botClients.length; i++)
            {
                rs[i] = botClients[i].gameStep(players[i]);
                envSteps[i]++;

                if (rs[i].done[0] || envSteps[i] >= maxSteps)
                {
                    // Save terminal values
                    for (int r = 0; r < terminalReward1.length; r++)
                    {
                        terminalReward1[r] = rs[i].reward[r];
                        terminalDone1[r] = rs[i].done[r];
                    }
                    // Reset and restore terminal values
                    botClients[i].reset(players[i]);
                    for (int r = 0; r < terminalReward1.length; r++)
                    {
                        rs[i].reward[r] = terminalReward1[r];
                        rs[i].done[r] = terminalDone1[r];
                    }
                    rs[i].done[0] = true;
                    envSteps[i] = 0;
                }
            }
            for (int i = 0; i < rs.length; i++)
            {
                reward[i] = rs[i].reward;
                done[i] = rs[i].done;
            }
            responses.set(null, reward, done);
            return responses;
        }

        // ---- Selfplay envs ----
        for (int i = 0; i < selfPlayClients.length; i++)
        {
            selfPlayClients[i].gameStep(actions[i * 2], actions[i * 2 + 1]);
            rs[i * 2]     = selfPlayClients[i].getResponse(0);
            rs[i * 2 + 1] = selfPlayClients[i].getResponse(1);
            envSteps[i * 2]++;
            envSteps[i * 2 + 1]++;

            if (rs[i * 2].done[0] || envSteps[i * 2] >= maxSteps)
            {
                // Save terminal rewards for both players
                for (int r = 0; r < terminalReward1.length; r++)
                {
                    terminalReward1[r] = rs[i * 2].reward[r];
                    terminalDone1[r]   = rs[i * 2].done[r];
                    terminalReward2[r] = rs[i * 2 + 1].reward[r];
                    terminalDone2[r]   = rs[i * 2 + 1].done[r];
                }
                // Reset and restore terminal values for both players
                selfPlayClients[i].reset();
                for (int r = 0; r < terminalReward1.length; r++)
                {
                    rs[i * 2].reward[r]     = terminalReward1[r];
                    rs[i * 2].done[r]       = terminalDone1[r];
                    rs[i * 2 + 1].reward[r] = terminalReward2[r];
                    rs[i * 2 + 1].done[r]   = terminalDone2[r];
                }
                rs[i * 2].done[0] = true;
                rs[i * 2 + 1].done[0] = true;
                envSteps[i * 2] = 0;
                envSteps[i * 2 + 1] = 0;
            }
        }

        // ---- Bot envs ----
        for (int i = selfPlayClients.length * 2; i < players.length; i++)
        {
            envSteps[i]++;
            rs[i] = clients[i - selfPlayClients.length * 2].gameStep(actions[i], players[i]);

            if (rs[i].done[0] || envSteps[i] >= maxSteps)
            {
                // Save terminal values
                for (int r = 0; r < rs[i].reward.length; r++)
                {
                    terminalReward1[r] = rs[i].reward[r];
                    terminalDone1[r] = rs[i].done[r];
                }
                // Reset and restore terminal values
                clients[i - selfPlayClients.length * 2].reset(players[i]);
                for (int r = 0; r < rs[i].reward.length; r++)
                {
                    rs[i].reward[r] = terminalReward1[r];
                    rs[i].done[r] = terminalDone1[r];
                }
                rs[i].done[0] = true;
                envSteps[i] = 0;
            }
        }

        // Collect into shared arrays
        for (int i = 0; i < rs.length; i++)
        {
            observation[i] = rs[i].observation;
            reward[i] = rs[i].reward;
            done[i] = rs[i].done;
        }
        responses.set(observation, reward, done);
        return responses;
    }

    // ------------------------------------------------------------------
    // Masks
    // ------------------------------------------------------------------

    /**
     * Collect valid action masks from all environments.
     *
     * Selfplay envs: each game provides masks for both players (P0 and P1).
     * Bot envs: masks for the RL agent's player only.
     *
     * @param player  which player the RL agent controls (for bot envs)
     * @return int[N_envs][H][W][mask_dim]
     */
    public int[][][][] getMasks(int player) throws Exception
    {
        for (int i = 0; i < selfPlayClients.length; i++)
        {
            masks[i * 2]     = selfPlayClients[i].getMasks(0);
            masks[i * 2 + 1] = selfPlayClients[i].getMasks(1);
        }
        for (int i = selfPlayClients.length * 2; i < masks.length; i++)
        {
            masks[i] = clients[i - selfPlayClients.length * 2].getMasks(player);
        }
        return masks;
    }

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    /** Dispose all GUI windows. Does NOT shut down the JVM. */
    public void close() throws Exception
    {
        if (clients != null)
        {
            for (JNIGridnetClient c : clients) { c.close(); }
        }
        if (selfPlayClients != null)
        {
            for (JNIGridnetClientSelfPlay c : selfPlayClients) { c.close(); }
        }
        if (botClients != null)
        {
            for (JNIBotClient c : botClients) { c.close(); }
        }
    }
}
