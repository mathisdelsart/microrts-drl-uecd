/*
 * Data container for vectorized (multi-env) step results.
 * Used by JNIGridnetVecClient to batch results from all environments.
 */
package ai.jni;

public class Responses
{
    public int[][][][] observation;  // [N_envs][13][H][W]
    public double[][] reward;        // [N_envs][N_reward_functions]
    public boolean[][] done;         // [N_envs][N_reward_functions]

    public Responses(int[][][][] observation, double[][] reward, boolean[][] done)
    {
        this.observation = observation;
        this.reward = reward;
        this.done = done;
    }

    /** Mutate in-place (avoids allocation every step). */
    public void set(int[][][][] observation, double[][] reward, boolean[][] done)
    {
        this.observation = observation;
        this.reward = reward;
        this.done = done;
    }
}
