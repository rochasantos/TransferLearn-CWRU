class PreprocessingPipeline:
    def __init__(self):
        self.steps = []

    def add_step(self, step):
        self.steps.append(step)

    def process(self, data, original_sr=None):        
        for step in self.steps:
            data = step.process(data, original_sr)
        return data
