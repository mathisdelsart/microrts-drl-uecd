package ai.reward;

import rts.GameState;
import rts.TraceEntry;

/**
 * Win/Draw/Loss reward that correctly handles draws (reward = 0.0).
 *
 * Standard WinLossRewardFunction returns -1.0 for both losses AND draws,
 * penalizing timeouts as if they were defeats. This version distinguishes:
 *   win  → +1.0
 *   loss → -1.0
 *   draw →  0.0
 *
 * From RAISocketAI (sgoodfriend).
 */
public class WinDrawLossRewardFunction extends RewardFunctionInterface
{
    public void computeReward(int maxplayer, int minplayer, TraceEntry te, GameState afterGs)
    {
        reward = 0.0;
        done = false;
        if (afterGs.gameover())
        {
            done = true;
            int winner = afterGs.winner();
            if (winner == maxplayer)      { reward = 1.0; }
            else if (winner == minplayer) { reward = -1.0; }
            else                          { reward = 0.0; }
        }
    }
}
