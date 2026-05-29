package ai.socket;

import rts.units.UnitTypeTable;

/**
 * Wrapper class to match bot name.
 * Delegates to UTS_Imass_SocketAI implementation.
 */
public class UTS_Imass extends UTS_Imass_SocketAI {
    
    public UTS_Imass(UnitTypeTable utt) {
        super(utt);
    }
    
    public UTS_Imass(int mt, int mi, String serverAddress, int port, int language, UnitTypeTable utt) {
        super(mt, mi, serverAddress, port, language, utt);
    }
}
