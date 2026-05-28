package ai.reward;

import rts.GameState;
import rts.TraceEntry;
import rts.UnitAction;
import rts.units.Unit;
import util.Pair;

/**
 * Ranged unit production reward: +1 per Ranged unit produced.
 *
 * From RAISocketAI (sgoodfriend).
 */
public class ProduceRangedUnitRewardFunction extends RewardFunctionInterface
{
    public void computeReward(int maxplayer, int minplayer, TraceEntry te, GameState afterGs)
    {
        reward = 0.0;
        done = false;
        for (Pair<Unit, UnitAction> p : te.getActions())
        {
            if (p.m_a.getPlayer() == maxplayer && p.m_b.getType() == UnitAction.TYPE_PRODUCE
                    && p.m_b.getUnitType() != null)
            {
                if (p.m_b.getUnitType().name.equals("Ranged")) { reward += 1; }
            }
        }
    }
}
