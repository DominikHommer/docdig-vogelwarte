class Module:
    module_key=None
    
    def __init__(self, module_key = None):
        if not module_key:
            raise Exception("Please declare a module key")
        
        self.module_key = module_key

    def get_preconditions(self) -> list[str]:
        raise Exception("Please define get_preconditions")
    
    def process(self, data: dict, config: dict) -> any:
        raise Exception("Please define process")
