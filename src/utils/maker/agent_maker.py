from .maker import Maker
from modules.agents.agent import Agent


class AgentMaker(Maker):
    """Factory class for creating Agents."""

    @staticmethod
    def make_rnn(*args, **kwargs) -> Agent:
        from modules.agents.rnn_agent import RNNAgent
        return RNNAgent(*args, **kwargs)    #TODO: Make old agents a subclass of Agent.

    @staticmethod
    def make_rnn_ns(*args, **kwargs) -> Agent:
        from modules.agents.rnn_ns_agent import RNNNSAgent
        return RNNNSAgent(*args, **kwargs)

    @staticmethod
    def make_rnn_feat(*args, **kwargs) -> Agent:
        from modules.agents.rnn_feature_agent import RNNFeatureAgent
        return RNNFeatureAgent(*args, **kwargs)

    @staticmethod
    def make_entity_attend_rnn(*args, **kwargs) -> Agent:
        """Entity agent using attention to merge obs."""
        from modules.agents.entity_attend_rnn_agent import EntityAttnRNNAgent
        return EntityAttnRNNAgent(*args, **kwargs)

    @staticmethod
    def make_entity_pooling_rnn(*args, **kwargs) -> Agent:
        """Entity agent using pooling to merge obs."""
        from modules.agents.entity_pooling_rnn_agent import EntityPoolingRNNAgent
        return EntityPoolingRNNAgent(*args, **kwargs)

    @staticmethod
    def make_pymarl2_qmix_agent(*args, **kwargs) -> Agent:
        """PyMARL2 QMix agent."""
        from modules.agents.pymarl2_qmix_agent import NRNNAgent
        return NRNNAgent(*args, **kwargs)

    @staticmethod
    def make_entity_catting_rnn(*args, **kwargs) -> Agent:
        """Entity agent using concatenation to merge obs."""
        from modules.agents.entity_catting_rnn_agent import EntityCattingRNNAgent
        return EntityCattingRNNAgent(*args, **kwargs)

    @staticmethod
    def make_entity_feature_attend_rnn(*args, **kwargs) -> Agent:
        """Entity agent using feature attention to merge obs."""
        from modules.agents.entity_feature_attend_rnn_agent import EntityFeatureAttnRNNAgent
        return EntityFeatureAttnRNNAgent(*args, **kwargs)

    @staticmethod
    def make_entity_maic(*args, **kwargs) -> Agent:
        """MAIC agent with entity scheme."""
        from modules.agents.entity_maic_agent import EntityMAICAgent
        return EntityMAICAgent(*args, **kwargs)

    @staticmethod
    def make_imagine_entity_attend_rnn_agent(*args, **kwargs) -> Agent:
        """Entity agent using feature attention to merge obs."""
        from modules.agents.imagine_entity_attend_rnn_agent import ImagineEntityAttnRNNAgent
        return ImagineEntityAttnRNNAgent(*args, **kwargs)

    @staticmethod
    def make_imagine_entity_attend_rnn_icm_agent(*args, **kwargs) -> Agent:
        """Entity agent using feature attention to merge obs."""
        from modules.agents.imagine_entity_attend_rnn_icm_agent import ImagineEntityAttnRNNAgentICM
        return ImagineEntityAttnRNNAgentICM(*args, **kwargs)

    @staticmethod
    def make_pymarl2_qmix_FiLMq_agent(*args, **kwargs) -> "FiLMAgent":
        """PyMARL2 QMix agent with FiLM."""
        from modules.agents.pymarl2_qmix_FiLMq_agent import FiLMAgent
        return FiLMAgent(*args, **kwargs)

    @staticmethod
    def make_pymarl2_qmix_FiLMq_agent_with_t(*args, **kwargs) -> "FiLMTAgent":
        """PyMARL2 QMix agent with FiLM."""
        from modules.agents.pymarl2_qmix_FiLMq_agent_with_t import FiLMTAgent
        return FiLMTAgent(*args, **kwargs)

    @staticmethod
    def make_pymarl2_qmix_DiT_FiLM(*args, **kwargs) -> "FiLMAgent":
        """PyMARL2 QMix agent with FiLM."""
        from modules.agents.pymarl2_qmix_DiT_FiLM import FiLMAgent
        return FiLMAgent(*args, **kwargs)

    @staticmethod
    def make_pymarl2_qmix_agent_with_t(*args, **kwargs) -> "NRNNAgent":
        """PyMARL2 QMix agent with T-Net."""
        from modules.agents.pymarl2_qmix_agent_with_t import NRNNAgent
        return NRNNAgent(*args, **kwargs)

    @staticmethod
    def make_entity_attend_FiLMq_agent(*args, **kwargs) -> "FiLMAgent":
        """Entity agent using attention to merge obs."""
        from modules.agents.entity_attend_FiLMq_agent import EntityFiLMAgent
        return EntityFiLMAgent(*args, **kwargs)

    @staticmethod
    def make_code_agent(*args, **kwargs) -> Agent:
        """Code agent using attention to merge obs."""
        from modules.agents.code_agent import CodeAgent
        return CodeAgent(*args, **kwargs)
    
    @staticmethod
    def make_kernel_agent(*args, **kwargs) -> Agent:
        """Code Kernel agent using attention to merge obs."""
        from modules.agents.kernel_agent import KernelAgent
        return KernelAgent(*args, **kwargs)

    @staticmethod
    def make_masia(*args, **kwargs) -> Agent:
        """MASIA agent with self-supervised message aggregation."""
        from modules.agents.masia_agent import MASIAAgent
        return MASIAAgent(*args, **kwargs)
 