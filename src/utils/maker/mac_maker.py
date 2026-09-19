from .maker import Maker
from controllers.mac import MAC


class MACMaker(Maker):
    """Factory class for creating Controllers."""

    @staticmethod
    def make_basic_mac(*args, **kwargs) -> MAC:
        from controllers.basic_controller import BasicMAC
        return BasicMAC(*args, **kwargs)

    @staticmethod
    def make_non_shared_mac(*args, **kwargs) -> MAC:
        from controllers.non_shared_controller import NonSharedMAC
        return NonSharedMAC(*args, **kwargs)    # TODO migrate NonSharedMAC to MAC

    @staticmethod
    def make_maddpg_mac(*args, **kwargs) -> MAC:
        from controllers.maddpg_controller import MADDPGMAC
        return MADDPGMAC(*args, **kwargs)   # TODO migrate NonSharedMAC to MAC

    @staticmethod
    def make_entity_mac(*args, **kwargs) -> MAC:
        from controllers.entity_controller import EntityMAC
        return EntityMAC(*args, **kwargs)

    @staticmethod
    def make_pymarl2_nmac_controller(*args, **kwargs) -> MAC:
        """MAC Implementation from PyMARL2"""
        from controllers.pymarl2_nmac_controller import NMAC
        return NMAC(*args, **kwargs)

    @staticmethod
    def make_entity_maic_mac(*args, **kwargs) -> MAC:
        from controllers.entity_maic_controller import EntityMAICMAC
        return EntityMAICMAC(*args, **kwargs)

    def make_imagine_mac(*args, **kwargs) -> MAC:
        from controllers.imagine_controller import ImagineMAC
        return ImagineMAC(*args, **kwargs)

    @staticmethod
    def make_icm_mac(*args, **kwargs) -> MAC:
        """MAC Implementation from PyMARL2"""
        from controllers.icm_controller import ICMMAC
        return ICMMAC(*args, **kwargs)

    @staticmethod
    def make_entity_state_as_obs_mac(*args, **kwargs) -> MAC:
        from controllers.entity_state_as_obs_controller import EntityStateAsObsMAC
        return EntityStateAsObsMAC(*args, **kwargs)

    @staticmethod
    def make_pymarl2_nmac_controller_with_t(*args, **kwargs) -> "NMAC":
        """MAC Implementation from PyMARL2"""
        from controllers.pymarl2_nmac_controller_with_t import NMAC
        return NMAC(*args, **kwargs)

    @staticmethod
    def make_new_basic_mac(*args, **kwargs) -> MAC:
        """A more efficient BasicMAC along with new q learner."""
        from controllers.new_basic_controller import BasicMAC
        return BasicMAC(*args, **kwargs)

    @staticmethod
    def make_code_mac(*args, **kwargs) -> MAC:
        """Code MAC"""
        from controllers.code_controller import CodeMAC
        return CodeMAC(*args, **kwargs)
    
    @staticmethod
    def make_kernel_mac(*args, **kwargs) -> MAC:
        """Code Kernel MAC"""
        from controllers.kernel_controller import KernelMAC
        return KernelMAC(*args, **kwargs)
    
    @staticmethod
    def make_bcrbc_mac(*args, **kwargs) -> MAC:
        """BC-RBC joint block-causal MAC."""
        from controllers.bcrbc_mac import BCRBCMAC
        return BCRBCMAC(*args, **kwargs)

    @staticmethod
    def make_rdc_mac(*args, **kwargs) -> MAC:
        """RDC Echo delay-compensating MAC."""
        from controllers.rdc_controller import RDCMAC
        return RDCMAC(*args, **kwargs)

    @staticmethod
    def make_masia_mac(*args, **kwargs) -> MAC:
        """MASIA communication MAC."""
        from controllers.masia_controller import MASIAMAC
        return MASIAMAC(*args, **kwargs)
