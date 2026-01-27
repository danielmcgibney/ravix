# environment.py

import inspect

class EvalEnvironment:
    def __init__(self, frame_depth=0, additional_globals=None):  # Line updated to include additional_globals
        frame = inspect.currentframe().f_back
        for _ in range(frame_depth):
            frame = frame.f_back
        self.locals = frame.f_locals
        self.globals = frame.f_globals
        if additional_globals:  # Line updated to use additional_globals
            self.globals.update(additional_globals)
        #print(f"Captured environment at frame depth {frame_depth}:")
        #print(f"Locals: {list(self.locals.keys())}")
        #print(f"Globals: {list(self.globals.keys())}")

    def eval(self, expr):
        #print(f"Evaluating '{expr}' in environment with globals: {list(self.globals.keys())}")
        return eval(expr, self.globals, self.locals)
