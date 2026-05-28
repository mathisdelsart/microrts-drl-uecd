package ai.reward;

import rts.GameState;
import rts.TraceEntry;
import rts.units.Unit;

/**
 * Army score reward based on unit cost weighted by remaining HP.
 *
 * Each unit's score = cost * (1 + HP / maxHP), so a full-health Heavy (cost 3)
 * scores 6.0 while a half-health Worker (cost 1) scores 1.5.
 *
 * reward = (ownScore - oppScore) / (ownScore + oppScore + 1)
 *
 * Normalized to roughly [-1, 1]. Provides a dense signal for the cost
 * value head (military advantage tracking).
 *
 * From RAISocketAI (sgoodfriend).
 */
public class MilitaryScoreRewardFunction extends RewardFunctionInterface
{
    public void computeReward(int maxplayer, int minplayer, TraceEntry te, GameState afterGs)
    {
        reward = 0.0;
        done = afterGs.gameover();
        double ownScore = 0;
        double oppScore = 0;
        for (Unit u : afterGs.getUnits())
        {
            double unitScore = u.getCost() * (1 + (double) u.getHitPoints() / u.getMaxHitPoints());
            if (u.getPlayer() == maxplayer)      { ownScore += unitScore; }
            else if (u.getPlayer() == minplayer) { oppScore += unitScore; }
        }
        reward = (ownScore - oppScore) / (ownScore + oppScore + 1);
    }
}
