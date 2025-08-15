from modules.module_base import Module
from dotenv import load_dotenv
import os

class Pipeline:
    """
    Stellt eine modulare Pipeline zusammen, in der verschiedene Verarbeitungsschritte
    (Klassen mit einer process()-Methode) sequentiell ausgeführt werden.
    """
    def __init__(self, input_data: dict = {}):
        self.stages: list[Module] = []
        self.data: dict = input_data
        self.config: dict = {}
    
    def add_stage(self, stage):
        self.stages.append(stage)

    def _setup_environment(self) -> bool:
        load_dotenv()

        if not os.path.exists("./config/detection_config.json"):
            print("[TATR Error]: ./config/detection_config.json missing")
            return False
        
        if not os.path.exists("./config/structure_config.json"):
            print("[TATR Error]: ./config/structure_config.json missing")
            return False
        
        if not os.path.exists("./config/pubtables1m_detection_detr_r18.pth"):
            print("[TATR Error]: ./config/pubtables1m_detection_detr_r18.pth missing. See link_to_tatr_models.txt")
            return False
        
        if not os.path.exists("./config/pubtables1m_structure_detr_r18.pth"):
            print("[TATR Error]: ./config/pubtables1m_structure_detr_r18.pth missing. See link_to_tatr_models.txt")
            return False
        
        if not os.path.exists("./config/denoise_model.keras"):
            print("[Denoise Error]: ./config/denoise_model.keras missing.")
            return False
        
        self.config = {
            'tatr': {
                'detection_config': './config/detection_config.json',
                'structure_config': './config/structure_config.json',
                'detection_model': './config/pubtables1m_detection_detr_r18.pth',
                'structure_model': './config/pubtables1m_structure_detr_r18.pth',
            },
            'denoise': {
                'model': './config/denoise_model.keras',
            },
        }

        return True

    def _check_condition(self, module: Module):
        preconditions = module.get_preconditions()
        if not any(self.data.get(cond) is not None for cond in preconditions):
            raise Exception(f"Preconditions for {module.module_key} not fulfilled: need one of {preconditions}")

    def run(self, input_data = None):
        if not self._setup_environment():
            raise Exception("Environment setup failed, please look inside logs for error")

        self.data['input'] = input_data
        for module in self.stages:
            self._check_condition(module)

            self.data[module.module_key] = module.process(self.data, self.config)

        # Output is result of last stage
        return self.data[self.stages[-1].module_key]