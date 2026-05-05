import ast
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ExtractedPDE:
    pde_fn_source: str = ""
    domain_type: str = ""
    spatial_bounds: list = field(default_factory=list)
    time_range: Optional[tuple] = None
    bcs: list = field(default_factory=list)
    ics: list = field(default_factory=list)
    num_domain: int = 5000
    num_boundary: int = 1000
    num_initial: int = 1000
    framework: str = "DeepXDE"

class DeepXDETranspiler(ast.NodeVisitor):
    def __init__(self): self.extracted = ExtractedPDE(framework="DeepXDE"); self._source_lines = []
    def parse(self, source: str): self._source_lines = source.splitlines(); tree = ast.parse(source); self.visit(tree); return self.extracted
    def visit_FunctionDef(self, node):
        if len(node.args.args) == 2: self.extracted.pde_fn_source = ast.get_source_segment("\n".join(self._source_lines), node) or ""
        self.generic_visit(node)
    def visit_Call(self, node):
        fn = self._get_call_name(node)
        if "GeometryXTime" in fn: self.extracted.domain_type = "spacetime"
        if "Interval" in fn and len(node.args) >= 2:
            lo, hi = self._eval_const(node.args[0]), self._eval_const(node.args[1])
            if lo is not None and hi is not None: self.extracted.spatial_bounds.append((lo, hi))
        if "TimeDomain" in fn and len(node.args) >= 2:
            t0, t1 = self._eval_const(node.args[0]), self._eval_const(node.args[1])
            if t0 is not None and t1 is not None: self.extracted.time_range = (t0, t1)
        if "DirichletBC" in fn: self.extracted.bcs.append({"type": "dirichlet"})
        if fn.endswith(".IC") or "icbc.IC" in fn: self.extracted.ics.append({"type": "initial"})
        for kw in node.keywords:
            if kw.arg == "num_domain": self.extracted.num_domain = int(self._eval_const(kw.value) or 5000)
        self.generic_visit(node)
    def _get_call_name(self, node): return node.func.attr if isinstance(node.func, ast.Attribute) else (node.func.id if isinstance(node.func, ast.Name) else "")
    def _eval_const(self, node): return node.value if isinstance(node, ast.Constant) else (-node.operand.value if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) else None)
    def to_rippl_script(self):
        bounds = self.extracted.spatial_bounds
        if self.extracted.time_range:
            bounds = [list(self.extracted.time_range)] + [list(b) for b in bounds]
        
        return f"""import rippl as rp
import rippl.nn as rnn

# Original Framework: {self.extracted.framework}
domain = rp.Domain(
    spatial_dims={len(bounds)},
    bounds={bounds},
    batch_size={self.extracted.num_domain}
)

model = rnn.MLP(in_dim={len(bounds)}, out_dim=1)
result = rp.run(domain=domain, equation=None, model=model)
"""

class ModulusTranspiler(ast.NodeVisitor):
    def __init__(self): self.extracted = ExtractedPDE(framework="modulus")
    def parse(self, s): self.visit(ast.parse(s)); return self.extracted
    def to_rippl_script(self): return "# rippl migrated from Modulus\nimport rippl as rp"

class SciANNTranspiler(ast.NodeVisitor):
    def __init__(self): self.extracted = ExtractedPDE(framework="sciann")
    def parse(self, s): self.visit(ast.parse(s)); return self.extracted
    def to_rippl_script(self): return "# rippl migrated from SciANN\nimport rippl as rp"

def migrate(source: str, framework: str = "auto") -> str:
    if framework == "auto": framework = _detect_framework(source)
    if framework == "deepxde": t = DeepXDETranspiler()
    elif framework == "modulus": t = ModulusTranspiler()
    elif framework == "sciann": t = SciANNTranspiler()
    else: raise MigrationError(f"Unsupported: {framework}")
    t.parse(source); return t.to_rippl_script()

def _detect_framework(s: str) -> str:
    if "deepxde" in s.lower(): return "deepxde"
    if "modulus" in s.lower(): return "modulus"
    if "sciann" in s.lower(): return "sciann"
    raise MigrationError("Unknown framework")

class MigrationError(ValueError): pass
