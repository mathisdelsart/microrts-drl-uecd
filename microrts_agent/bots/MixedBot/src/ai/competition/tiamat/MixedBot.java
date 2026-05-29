package ai.competition.tiamat;

import rts.units.UnitTypeTable;

/**
 * Wrapper class to match bot name.
 * Delegates to ImprovedTiamat implementation.
 */
public class MixedBot extends ImprovedTiamat {
    
    public MixedBot(UnitTypeTable utt) {
        super(utt);
    }
}
