/*
 * Data container for a single env's step result.
 * Reused via set() to avoid allocation per step.
 */
package ai.jni;

public class Response
{
    public int[][][] observation;  // [13][H][W] raw features
    public double[] reward;        // [N_reward_functions]
    public boolean[] done;         // [N_reward_functions]
    public String info;            // extra info (unused)

    public Response(int[][][] observation, double[] reward, boolean[] done, String info)
    {
        this.observation = observation;
        this.reward = reward;
        this.done = done;
        this.info = info;
    }

    /** Mutate in-place (avoids allocation every step). */
    public void set(int[][][] observation, double[] reward, boolean[] done, String info)
    {
        this.observation = observation;
        this.reward = reward;
        this.done = done;
        this.info = info;
    }
}
