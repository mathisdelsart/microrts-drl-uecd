/*
 * Interface that all JNI agents must implement.
 * Defines the contract between the game engine and Python.
 */
package ai.jni;

import rts.GameState;
import rts.PlayerAction;

public interface JNIInterface
{
    /** Convert int[][] action array from Python into a PlayerAction. */
    PlayerAction getAction(int player, GameState gs, int[][] actions) throws Exception;

    /** Extract raw obs int[13][H][W] from the game state. */
    int[][][] getObservation(int player, GameState gs) throws Exception;

    /** Reset internal state between episodes. */
    void reset();

    /** Compute reward delta (legacy: not used, we use RewardFunctionInterface[]). */
    double computeReward(int maxplayer, int minplayer, GameState gs) throws Exception;

    /** Return extra info string (unused in practice). */
    String computeInfo(int player, GameState gs) throws Exception;
}
