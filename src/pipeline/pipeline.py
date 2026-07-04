from modules.module_base import Module
from dotenv import load_dotenv
import os
import traceback

class Pipeline:
    """
    Stellt eine modulare Pipeline zusammen, in der verschiedene Verarbeitungsschritte
    (Klassen mit einer process()-Methode) sequentiell ausgeführt werden.
    """
    def __init__(self, input_data: dict = None):
        self.stages: list[Module] = []
        # Avoid the classic mutable-default-arg trap that would have every
        # Pipeline instance share the same dict.
        self.data: dict = dict(input_data) if input_data else {}
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
        
        # Note: denoise_model.keras is no longer required — CellDenoiser was
        # dropped from the default pipeline because it hurt downstream OCR.
        # The check is left as a soft warning so legacy setups keep working
        # if someone wires CellDenoiser back in.
        if not os.path.exists("./config/denoise_model.keras"):
            print("[Pipeline] Note: ./config/denoise_model.keras missing — fine unless you re-enable CellDenoiser.")

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

    def _is_critical(self, module: Module) -> bool:
        """Pre-processing stages whose failure makes the rest pointless."""
        return module.module_key in {
            "pdf-converter",
            "table-rotator",
            "tatr-extractor",
            "merged-column-extractor",
            "row-extractor",  # MergedRowExtractor registers under this key
        }

    def run(self, input_data=None):
        if not self._setup_environment():
            raise Exception("Environment setup failed, please look inside logs for error")

        self.data['input'] = input_data

        for module in self.stages:
            critical = self._is_critical(module)
            try:
                self._check_condition(module)
            except Exception as e:
                msg = f"[Pipeline] Skipping {module.module_key}: {e}"
                if critical:
                    raise
                print(msg)
                continue

            try:
                self.data[module.module_key] = module.process(self.data, self.config)
            except Exception as e:
                if critical:
                    raise
                print(f"[Pipeline] Module {module.module_key} crashed: {e}")
                if os.environ.get("DOCDIG_TRACEBACK"):
                    traceback.print_exc()
                # Leave self.data[module.module_key] unset so downstream modules
                # fall back to earlier inputs via their precondition list.

        # Output is result of the last stage that actually produced something,
        # falling back through the pipeline tail so the caller gets the most
        # complete view available.
        for module in reversed(self.stages):
            if module.module_key in self.data and self.data[module.module_key] is not None:
                return self.data[module.module_key]
        return None