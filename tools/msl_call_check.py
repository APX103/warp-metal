"""Static call-signature net for translated MSL units (KEP-0001 M3.2).

Every prelude-provided helper (``wp_*`` atomics, ``make_*`` constructors,
``wp_extract``/``wp_vec_insert``/...) is resolved at Metal compile time by
overload lookup. A translator that emits a well-formed call text still fails
loud at the first ``wp.build_source`` if no overload matches — the failure
mode the M3.2 body-force battery surfaced (``wp_atomic_add_ret_at`` on a
2-D vec3f array with no matching overload).

This tool closes that gap host-side, before any Metal compile:

1. *Prototype DB* — parsed from the prelude plus every per-unit generated
   definition (``ensure_vec_def``/``ensure_mat_def`` output), so name, arity
   and parameter types are known for everything the compile unit provides.
2. *Call extraction* — every ``wp_*``/``make_*`` call site in the kernel and
   its transitively translated ``@wp.func`` bodies.
3. *Rough type-compat check* — pointer base types, scalar names, struct
   value types. Literal ints are accepted wherever any integer (or float)
   parameter is expected (MSL implicit conversion); unresolvable arguments
   (nested arithmetic) demote to warnings instead of failures.

What the checker CANNOT catch, honestly:
- overload resolution subtleties Metal applies that our strict-matching
  would also accept (e.g. identical arity with implicit scalar conversions
  we allow by design);
- address-space mismatches beyond the device-pointer convention
  (``thread`` vs ``device``) inside a matching base type;
- semantic bugs: a resolving overload can still be the wrong one
  (1-D vs 2-D index math) — arity plus pointer type narrows but does not
  prove intent;
- anything outside ``wp_*``/``make_*`` calls: user ``@wp.func`` resolution
  is the translator's own job and fails there.

Two phases, matching the corpus two-venv seam (the translator only accepts
kernels defined under the vendored warp, and ground truth needs real warp):

``--kernel module:attr``   translate one kernel (attr may be a factory call
like ``mujoco_warp._src.smooth:_subtree_com_acc(False)``) and check it.
Use under the shim.

``--scenario body_force``  phase A: replay the Newton body-force battery
scenarios under REAL warp on CPU and print the exact launch closure —
every (kernel, argument-shape) pair plus the set of defining modules.
Factory-built kernels invisible to ``dir()`` sweeps show up here. Refuses
to run under the metal shim; imports of ``warp_metal`` must come after it.

``--sweep MOD[,MOD...]``   phase B: import the closure modules under the
shim and translate + check every kernel reachable via module attributes
plus ``@cache_kernel``-style factories called with small bool/int grids.
Nothing launches and nothing compiles: kernel construction is pure
codegen. Diff the phase A kernel list against this sweep to confirm
closure coverage (module-level sweep is a superset).

Exit code is nonzero when any ERROR finding exists.
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys

# ---------------------------------------------------------------------------
# type names
# ---------------------------------------------------------------------------

_RE_VEC_T = re.compile(r"^wp::vec_t<\s*(\d+)\s*,\s*wp::(float32|int32)\s*>$")
_RE_MAT_T = re.compile(r"^wp::mat_t<\s*(\d+)\s*,\s*(\d+)\s*,\s*wp::(float32|int32)\s*>$")


def normalize(t: str) -> str:
    """Canonicalize an MSL/warp type name for comparison."""
    t = t.strip()
    t = re.sub(r"^\s*(device|thread|constant|threadgroup)\s+", "", t)
    t = t.rstrip("*&").strip()
    m = _RE_VEC_T.match(t)
    if m:
        n, elem = m.groups()
        return f"vec{n}{'f' if elem == 'float32' else 'i'}"
    m = _RE_MAT_T.match(t)
    if m:
        r, c, elem = m.groups()
        return f"mat{r}{'x' + c if r != c else ''}{'f' if elem == 'float32' else 'i'}"
    return t


# ---------------------------------------------------------------------------
# prototype DB
# ---------------------------------------------------------------------------

_RE_PROTO = re.compile(r"(?:template\s*<[^>]*>\s*)?static inline\s+([A-Za-z_][\w:<> ,*&]*?)\s+(wp_\w+|make_\w+)\s*\(([^{;]*?)\)\s*\{")

_INT_NAMES = {"int", "uint", "int64_t", "uint64_t", "long", "size_t"}
_SCALAR_NAMES = _INT_NAMES | {"float", "bool", "uchar"}


class Proto:
    __slots__ = ("name", "params")

    def __init__(self, name, params):
        self.name = name
        self.params = params  # [(normalized base type, is_pointer)]


def _split_params(blob: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in blob:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


class PreludeFn:
    """Stub of an MSLFunction for check_function: one prelude definition."""

    __slots__ = ("name", "params")

    def __init__(self, name, params):
        self.name = name
        self.params = params  # [SimpleNamespace(kind, msl, name, array_fields)]


def _parse_param(p: str, index: int):
    """One prototype parameter -> (kind, msl type, param name)."""
    p = p.strip()
    if not p:
        return None
    # `device T*` binds a device pointer (devptr); `thread T&` binds a
    # reference the caller passes by value syntax (value); `device T&`
    # binds a dereferenced DEVICE pointer — callers write
    # `*wp_address(...)` (the inline component_ref store target,
    # KEP-0001 M3.2), which classifies ptr:T
    m_space = re.match(r"^(?:const\s+)?(device|thread|constant|threadgroup)\s+", p)
    space = m_space.group(1) if m_space else None
    is_ptr = "*" in p or (space in ("device", "constant", "threadgroup") and "&" in p)
    base = re.sub(r"^\s*(?:const\s+)?(?:device|thread|constant|threadgroup)\s+", "", p)
    base = base.replace("*", "").replace("&", "").strip()
    tokens = base.split()
    # strip a trailing parameter *name*: "int i", "vec3f v",
    # "wp_arr_meta m", "T data" — the last bare identifier is the name
    # whenever the first token starts a known type vocabulary
    pname = f"arg{index}"
    if (
        len(tokens) >= 2
        and re.match(r"^[A-Za-z_]\w*$", tokens[-1])
        and (
            tokens[0] in _SCALAR_NAMES
            or tokens[0] == "wp_arr_meta"
            or "::" in tokens[0]
            or re.match(r"^(vec\w*|mat\w*|quat\w*|spatial_\w*|transform\w*|atomic_\w*|wp\w*|int\w*|uint\w*|T)$", tokens[0])
        )
    ):
        pname = tokens[-1]
        base = " ".join(tokens[:-1])
    base = normalize(base)
    if base == "void":
        base = "int"
    return ("devptr" if is_ptr else "value", base, pname)


def parse_functions(src: str) -> list:
    """Split a prelude-like source into (PreludeFn, body_lines) pairs.

    Uses the prototype regex to find each definition, then brace-balances
    the body. The prototype parameter view and the DB view share this
    parser, so they can never drift.
    """
    from types import SimpleNamespace

    out = []
    for m in _RE_PROTO.finditer(src):
        params = []
        for p in _split_params(m.group(3)):
            parsed = _parse_param(p, len(params))
            if parsed is not None:
                params.append(SimpleNamespace(kind=parsed[0], msl=parsed[1], name=parsed[2], array_fields=[]))
        # brace-balance the body from the end of the prototype match
        i = m.end()
        depth = 1
        while depth and i < len(src):
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
            i += 1
        body = src[m.end() : i - 1].splitlines()
        out.append((PreludeFn(m.group(2), params), body))
    return out


def parse_prototypes(src: str) -> dict[str, list[Proto]]:
    """Parse ``static inline`` definitions into (name -> [Proto])."""
    db: dict[str, list[Proto]] = {}
    for fn, _body in parse_functions(src):
        params = [(p.msl, p.kind == "devptr") for p in fn.params]
        db.setdefault(fn.name, []).append(Proto(fn.name, params))
    return db


# ---------------------------------------------------------------------------
# call extraction + classification
# ---------------------------------------------------------------------------

_RE_CALL = re.compile(r"\b(wp_[A-Za-z0-9_]+|make_[A-Za-z0-9_]+)\s*\(")
_RE_DECL = re.compile(r"^\s*(?:const\s+)?([A-Za-z_][\w:]*(?:\s*<[^;=]*?>)?)\s+([A-Za-z_]\w*)\s*=")
_RE_FOR_DECL = re.compile(r"\bfor\s*\(\s*(int|uint|float)\s+([A-Za-z_]\w*)\s*=")
_RE_LIT_INT = re.compile(r"^-?\d+$|^-?0[xX][0-9a-fA-F]+$")
_RE_LIT_FLOAT = re.compile(r"^-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE]-?\d+)?f?$")
_KEYWORDS = {"if", "for", "while", "switch", "return", "else", "sizeof"}


class Finding:
    __slots__ = ("severity", "unit", "line", "message")

    def __init__(self, severity, unit, line, message):
        self.severity = severity  # 'ERROR' | 'WARN'
        self.unit = unit
        self.line = line
        self.message = message

    def __repr__(self):
        return f"{self.severity} [{self.unit}:{self.line}] {self.message}"


def _extract_calls(lines):
    """Yield (name, [arg strings], lineno) for every wp_*/make_* call.

    Operates on the JOINED body text so calls split across lines parse with
    correct arity; lineno is the line of the call's opening name.
    """
    text = "\n".join(lines)
    for m in _RE_CALL.finditer(text):
        depth, end = 0, m.end()
        for i in range(m.end() - 1, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        args = _split_params(text[m.end() : end])
        lineno = text.count("\n", 0, m.start()) + 1
        yield m.group(1), args, lineno


def self_check_prelude(src=None):
    """Check every prelude function's INTERNAL calls against the DB.

    The kernel-side net scans kernel + @wp.func bodies; a missing overload
    INSIDE the prelude itself (one family delegating to a scalar core that
    was never defined) compiles into every unit and fails the whole
    battery. This pass walks each prelude definition body instead.
    """
    from warp_metal.msl.prelude import PRELUDE as _PRELUDE

    if src is None:
        src = _PRELUDE
    db = parse_prototypes(src)
    findings = []
    for fn, body in parse_functions(src):
        if not any(_RE_CALL.search(line) for line in body):
            continue
        findings += UnitChecker(db).check_function(f"prelude:{fn.name}", fn, body)
    return findings


class UnitChecker:
    """Check one KernelUnit against a prototype DB."""

    def __init__(self, proto_db):
        self.db = proto_db

    def _collect_maps(self, fn, args=None):
        """Data-pointer map ({name}_data -> elem) and value-var map."""
        data_map, var_map = {}, {}
        sources = []
        if args:
            for a in args:
                try:
                    msl = a.msl_elem if a.kind == "array" else a.msl
                except Exception:  # noqa: BLE113 - foreign ctype (wp::Any etc.)
                    msl = "unknown"
                sources.append([(a.kind, a.name, msl, getattr(a, "array_fields", []) or [])])
        for p in getattr(fn, "params", []) or []:
            sources.append([(p.kind, p.name, p.msl, p.array_fields or [])])
        for src in sources:
            for kind, name, msl, fields in src:
                if kind == "array":
                    data_map[f"{name}_data"] = normalize(msl)
                    var_map[name] = ("arr", normalize(msl))
                elif kind == "devptr":
                    # bare device-pointer parameter (prelude function bodies
                    # pass pointers under their plain param names)
                    data_map[name] = normalize(msl)
                elif kind == "struct":
                    for field, elem in fields:
                        data_map[f"{name}__{field}_data"] = normalize(elem)
                else:
                    var_map[name] = ("val", normalize(msl))
        return data_map, var_map

    def _classify(self, expr, data_map, var_map):
        e = expr.strip()
        # unwrap outer parentheses — only when they truly wrap the whole
        # expression (the matching close of e[0] must be the last char),
        # so cast-then-operand chains `(atomic_float*)(x + y)` survive
        while e.startswith("("):
            depth, end = 0, -1
            for i, ch in enumerate(e):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end == len(e) - 1:
                e = e[1:-1].strip()
            else:
                break
        # explicit casts decide the type: `(device atomic_float*)(...)` is
        # pointer-to-atomic_float however the operand started; `(uint64_t)x`
        # is a uint64_t value. This is how the prelude legally bridges plain
        # element pointers to the atomic cores.
        if m := re.match(r"^\(\s*(?:const\s+)?(?:(?:device|thread|constant|threadgroup)\s+)?([\w:]+)\s*\*\s*\)", e):
            return ("ptr", normalize(m.group(1)))
        if m := re.match(r"^\(\s*(?:const\s+)?([\w:]+)\s*\)\s*", e):
            t = normalize(m.group(1))
            if t in _SCALAR_NAMES:
                return ("val", t)
        if re.search(r"(==|!=|<=|>=|<|>|&&|\|\|)", e) and "(" not in e and ")" not in e:
            return ("val", "bool")
        if e in data_map:
            return ("ptr", data_map[e])
        if e.endswith("_data"):
            base = e[:-5]
            if "__" in base:
                return ("unknown", e)
            return ("unknown", e)
        if e.endswith("_m") or re.match(r"^_wp_args\.\w+_m$", e):
            return ("meta", None)
        if _RE_LIT_INT.match(e):
            return ("lit_int", None)
        if _RE_LIT_FLOAT.match(e):
            return ("lit_float", None)
        if e in var_map:
            return var_map[e]
        if e.startswith("_wp_args."):
            member = e[len("_wp_args.") :]
            if member == "_dims":
                # translator-generated launch-dims meta in Params_*
                return ("meta", None)
            if member in var_map:
                return var_map[member]
            return ("unknown", e)
        if re.match(r"^\d+$", e):
            return ("lit_int", None)
        # pointer arithmetic / address-cast chains preserve the base type:
        # `c + 0`, `(device atomic_float*)((device char*)d + ...)`, `&v[i]` —
        # if any identifier resolves to a known device pointer, the result
        # is that pointer
        for ident in re.findall(r"[A-Za-z_]\w*", e):
            if ident in data_map:
                return ("ptr", data_map[ident])
        return ("unknown", e)

    def check_function(self, unit_name, fn, lines, args=None):
        findings = []
        # strip // comments first: they carry helper-name mentions that
        # would otherwise register as phantom call sites
        lines = [line.split("//", 1)[0] for line in lines]
        data_map, var_map = self._collect_maps(fn, args)
        # translator-generated kernel preamble locals (kernel_src emits them
        # outside fn.lines): scalar thread index, tile lane, fold guard
        for name_ in ("_idx", "_wp_lane", "_wp_live"):
            var_map.setdefault(name_, ("val", "int"))
        # typed decls in the body override/augment the param maps; pointer
        # decls (`device float* x_data = ...`, phi-materialized locals) feed
        # the data-pointer map instead
        for line in lines:
            m = _RE_DECL.match(line)
            if m and m.group(1) not in _KEYWORDS:
                decl_t = m.group(1).strip()
                if decl_t.endswith("*"):
                    data_map[m.group(2)] = normalize(decl_t[:-1])
                else:
                    var_map[m.group(2)] = ("val", normalize(decl_t))
            for fm in _RE_FOR_DECL.finditer(line):
                var_map[fm.group(2)] = ("val", normalize(fm.group(1)))
        for name, args_, ln_no in _extract_calls(lines):
            protos = self.db.get(name)
            if protos is None:
                findings.append(
                    Finding("ERROR", unit_name, ln_no, f"{name}: no prototype found in prelude or generated defs")
                )
                continue
            got = [self._classify(a, data_map, var_map) for a in args_]
            compatible = []
            for proto in protos:
                if len(proto.params) != len(got):
                    continue
                ok = True
                for (pbase, pptr), (kind, val) in zip(proto.params, got):
                    if pbase == "T":
                        # template <typename T> generic: any pointer/value
                        continue
                    if pbase == "wp_arr_meta":
                        # kernel sites pass `..._m`; prelude sites pass the
                        # wp_arr_meta parameter directly by name
                        if kind == "meta" or (kind == "val" and val == "wp_arr_meta"):
                            continue
                        ok = False
                        break
                    if pptr:
                        if kind != "ptr" or normalize(val) != pbase:
                            ok = False
                            break
                    elif pbase in _SCALAR_NAMES:
                        if kind == "lit_int" and (pbase in _INT_NAMES or pbase == "float"):
                            continue
                        if kind == "lit_float" and pbase == "float":
                            continue
                        if kind == "val" and val == pbase:
                            continue
                        # integer-family implicit conversions are legal MSL
                        # (uint dims vars into int index params, uint64 into
                        # int, int into uint); bool accepts any scalar
                        if pbase in _INT_NAMES and kind == "val" and val in _INT_NAMES:
                            continue
                        if pbase == "bool" and kind == "val" and val in _SCALAR_NAMES:
                            continue
                        ok = False
                        break
                    else:
                        if kind == "val" and val == pbase:
                            continue
                        ok = False
                        break
                if ok:
                    compatible.append(proto)
            if not compatible:
                got_desc = ", ".join(f"{kind}:{val}" if val is not None else kind for kind, val in got)
                arity = [p for p in protos if len(p.params) == len(got)]
                # findings with unresolvable argument positions demote to
                # WARN (struct member chains, foreign ctypes): the checker
                # cannot prove them either way
                unresolvable = any(kind == "unknown" for kind, _ in got)
                sev = "WARN" if unresolvable else "ERROR"
                if arity:
                    exp = " | ".join(",".join(f"{'ptr ' + b if ip else b}" for b, ip in p.params) for p in arity)
                    findings.append(
                        Finding(
                            sev,
                            unit_name,
                            ln_no,
                            f"{name} arity {len(got)} no type match: args ({got_desc}); expected ({exp})",
                        )
                    )
                else:
                    arities = sorted({len(p.params) for p in protos})
                    findings.append(
                        Finding(
                            sev,
                            unit_name,
                            ln_no,
                            f"{name}: no overload with arity {len(got)} (known arities: {arities})",
                        )
                    )
        return findings


def check_unit(unit, proto_db=None) -> list[Finding]:
    """Check a KernelUnit: prelude + generated defs vs all call sites."""
    from warp_metal.msl.prelude import PRELUDE

    if proto_db is None:
        proto_db = parse_prototypes(PRELUDE)
        # generated defs (ensure_vec_def / ensure_mat_def output) live here
        for definition in unit.structs.values():
            for name, protos in parse_prototypes(definition).items():
                proto_db.setdefault(name, []).extend(protos)
    checker = UnitChecker(proto_db)
    findings = []
    findings += checker.check_function(unit.kernel.name, unit.kernel, unit.kernel.lines, unit.args)
    for fn in unit.functions:
        findings += checker.check_function(f"{unit.kernel.name}/{fn.name}", fn, fn.lines)
    return findings


# ---------------------------------------------------------------------------
# CLI: --kernel mode
# ---------------------------------------------------------------------------

def _load_kernel(spec: str):
    """module:attr[factory args] -> vendored Kernel object."""
    mod_name, expr = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    return eval(expr, vars(mod))  # noqa: S307 - trusted tool input


def run_kernel_mode(specs):
    from warp_metal.msl.translate import TranslationError, translate_kernel

    total_err = 0
    for spec in specs:
        try:
            kernel = _load_kernel(spec)
            unit = translate_kernel(kernel)
        except TranslationError as e:
            print(f"TRANSLATE_FAIL [{spec}] {e}")
            total_err += 1
            continue
        except Exception as e:  # noqa: BLE113 - report and continue
            print(f"BUILD_FAIL [{spec}] {type(e).__name__}: {e}")
            total_err += 1
            continue
        findings = check_unit(unit)
        errs = [f for f in findings if f.severity == "ERROR"]
        total_err += len(errs)
        if not findings:
            print(f"OK [{spec}]")
            continue
        # group repeated sites (same finding text with different line numbers)
        groups: dict[str, list[Finding]] = {}
        for f in findings:
            key = f.message.split(": args")[0] if "args" in f.message else f.message
            groups.setdefault(key, []).append(f)
        for key, fs in groups.items():
            sev = "ERROR" if any(f.severity == "ERROR" for f in fs) else "WARN"
            lines_desc = f"x{len(fs)} @lines {fs[0].line}+" if len(fs) > 1 else f"@line {fs[0].line}"
            msg = fs[0].message.split("args")[0] if len(fs) > 1 and "args" in fs[0].message else fs[0].message
            print(f"{sev} [{spec} {lines_desc}] {msg}")
    return total_err


# ---------------------------------------------------------------------------
# CLI: --scenario mode (real warp CPU closure trace)
# ---------------------------------------------------------------------------

_TRACE_SCENARIOS = ("body_force",)


def _run_body_force_trace():
    """Replay newton.tests.test_body_force mujoco_warp scenarios on CPU."""
    import unittest

    import newton
    import newton.tests.test_body_force as tbf

    class _Harness(unittest.FunctionTestCase):
        pass

    harness = _Harness(lambda: None)

    mujoco_warp_fn = lambda model: newton.solvers.SolverMuJoCo(  # noqa: E731
        model, use_mujoco_cpu=False, disable_contacts=True
    )
    mujoco_warp_rk4_fn = lambda model: newton.solvers.SolverMuJoCo(  # noqa: E731
        model, integrator="rk4", use_mujoco_cpu=False, disable_contacts=True
    )

    cases = [
        (tbf.test_floating_body, dict(solver_fn=mujoco_warp_fn, test_angular=False)),
        (tbf.test_floating_body, dict(solver_fn=mujoco_warp_fn, test_angular=True)),
        (tbf.test_floating_body, dict(solver_fn=mujoco_warp_fn, test_angular=False, use_control=True)),
        (tbf.test_3d_articulation, dict(solver_fn=mujoco_warp_fn, test_angular=True, up_axis=newton.Axis.Y)),
        (tbf.test_3d_articulation, dict(solver_fn=mujoco_warp_fn, test_angular=False, up_axis=newton.Axis.Y)),
        (tbf.test_force_no_rotation, dict(solver_fn=mujoco_warp_rk4_fn)),
        (tbf.test_force_no_rotation, dict(solver_fn=mujoco_warp_rk4_fn, use_control=True)),
        (tbf.test_combined_force_torque, dict(solver_fn=mujoco_warp_rk4_fn)),
        (tbf.test_combined_force_torque, dict(solver_fn=mujoco_warp_rk4_fn, use_control=True)),
    ]
    for fn, kw in cases:
        try:
            fn(harness, "cpu", **kw)
        except AssertionError:
            pass  # numeric outcomes are irrelevant; launches already happened


def _patch_launches(on_launch):
    """Intercept wp.launch / wp.launch_tiled during the trace."""
    import warp as wp

    real_launch = wp.launch
    real_tiled = getattr(wp, "launch_tiled", None)

    def wrap(kernel, dim=None, inputs=(), outputs=(), *a, **k):
        on_launch(kernel, inputs, outputs)
        return real_launch(kernel, dim=dim, inputs=inputs, outputs=outputs, *a, **k)

    wp.launch = wrap
    if real_tiled is not None:

        def wrap_tiled(kernel, dim=None, inputs=(), outputs=(), *a, **k):
            on_launch(kernel, inputs, outputs)
            return real_tiled(kernel, dim=dim, inputs=inputs, outputs=outputs, *a, **k)

        wp.launch_tiled = wrap_tiled
    return real_launch, real_tiled


def _arg_shapes(inputs, outputs):
    """Normalized (dtype-last-name, ndim) per launch argument."""
    import warp as wp

    def dt_key(dt):
        name = getattr(dt, "_wp_name__", None) or str(dt)
        m = re.search(r"(\w+)$", name)  # last identifier run (strip <class '...'>)
        return m.group(1) if m else name

    shapes = []
    for a in [*inputs, *outputs]:
        if isinstance(a, wp.array):
            shapes.append((dt_key(a.dtype), a.ndim))
        else:
            shapes.append((type(a).__name__, 0))
    return tuple(shapes)


def _closure_key(kernel, inputs, outputs):
    """Phase-portable dedup key: (kernel name, sorted normalized arg shapes).

    Phase A (real warp) and phase B (shim) produce comparable keys because
    dtype names are normalized to their last dotted component; the only
    cross-phase divergence is mjw's own composite types (vec10f), which keep
    their mujoco_warp._src.types home under both warps.
    """
    name = getattr(kernel, "__name__", str(kernel))
    try:
        shapes = _arg_shapes(inputs, outputs)
    except Exception:  # noqa: BLE113 - unshaped launch: dedup by name only
        return (name, None)
    return (name, tuple(sorted(shapes)))


def _trace_scenario_closure(scenario):
    """Phase A: run the scenario under REAL warp on CPU, print the closure.

    Ground truth for which kernels actually launch (data-dependent control
    flow included). Kernels are NOT translated here — the translator only
    accepts shim-defined kernels (vendored annotation objects), so phase B
    re-captures them under the shim.
    """
    import warp as wp

    warp_file = getattr(wp, "__file__", "") or ""
    if "warp_metal" in warp_file:
        raise SystemExit(
            "msl_call_check: sys.modules['warp'] is warp_metal — refusing to "
            "trace under the metal shim. Import real warp before warp_metal "
            "(e.g. `import warp` first) and unset NEWTON_WARP_BACKEND."
        )

    seen = set()
    order = []
    modules = set()

    def on_launch(kernel, inputs, outputs):
        key = _closure_key(kernel, inputs, outputs)
        if key not in seen:
            seen.add(key)
            order.append(key)
            modules.add(getattr(kernel, "__module__", "?") or "?")

    real_launch, real_tiled = _patch_launches(on_launch)
    try:
        if scenario == "body_force":
            _run_body_force_trace()
        else:
            raise SystemExit(f"unknown scenario {scenario!r}; known: {_TRACE_SCENARIOS}")
    finally:
        wp.launch = real_launch
        if real_tiled is not None:
            wp.launch_tiled = real_tiled

    for mod in sorted(modules):
        print(f"CLOSURE_MOD {mod}")
    for key in order:
        name, shapes = key
        shape_desc = ",".join(f"{d}x{n}" for d, n in shapes) if shapes else "?"
        print(f"CLOSURE {name} [{shape_desc}]")
    return 0


def _check_one(kernel, label, proto_db):
    """Translate + call-check one kernel; returns (status, error_count)."""
    from warp_metal.msl.translate import TranslationError, translate_kernel

    try:
        unit = translate_kernel(kernel)
    except TranslationError as e:
        print(f"TRANSLATE_FAIL [{label}] {type(e).__name__}: {str(e)[:200]}")
        return ("translate_fail", 1)
    except Exception as e:  # noqa: BLE113 - gap report
        print(f"TRANSLATE_FAIL [{label}] {type(e).__name__}: {str(e)[:200]}")
        return ("translate_fail", 1)
    local_db = dict((k, list(v)) for k, v in proto_db.items())
    for definition in unit.structs.values():
        for name, protos in parse_prototypes(definition).items():
            local_db.setdefault(name, []).extend(protos)
    findings = UnitChecker(local_db).check_function(label, unit.kernel, unit.kernel.lines, unit.args)
    for fn in unit.functions:
        findings += UnitChecker(local_db).check_function(f"{label}/{fn.name}", fn, fn.lines)
    errs = [f for f in findings if f.severity == "ERROR"]
    # group repeated sites (same finding head text)
    groups: dict[str, list[Finding]] = {}
    for f in findings:
        key = f.message.split(": args")[0] if "args" in f.message else f.message
        groups.setdefault(key, []).append(f)
    if not findings:
        return ("ok", 0)
    for key, fs in groups.items():
        sev = "ERROR" if any(f.severity == "ERROR" for f in fs) else "WARN"
        lines_desc = f"x{len(fs)} @lines {fs[0].line}+" if len(fs) > 1 else f"@line {fs[0].line}"
        msg = fs[0].message.split("args")[0] if len(fs) > 1 and "args" in fs[0].message else fs[0].message
        print(f"{sev} [{label} {lines_desc}] {msg}")
    return ("findings", len(errs))


_RE_KERNEL_DECOR = re.compile(r"@(\w+\.)?kernel\b")


def _factory_grid(func):
    """Small arg grid for a kernel factory: bools x {False,True}, ints x {1}.

    Product capped at 16 combos. Factory calls are cache_kernel wrappers
    (build + cache a Kernel) — safe to call with flag combos; anything that
    raises is skipped.
    """
    import inspect
    import itertools

    target = getattr(func, "__wrapped__", func)
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return [()]
    choices = []
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        ann = p.annotation
        if p.default is False or ann is bool or p.default is True:
            choices.append((False, True))
        elif p.default is not p.empty and isinstance(p.default, int) and not isinstance(p.default, bool):
            choices.append((p.default,))
        elif ann is int:
            choices.append((1,))
        else:
            choices.append((p.default,) if p.default is not p.empty else (1,))
    if not choices or len(choices) > 4:
        return [()]
    grid = list(itertools.product(*choices))
    return grid[:16]


def run_sweep_mode(module_names):
    """Sweep closure modules under the shim: every module-level kernel plus
    every kernel-building factory (bool/int flag grid), translated and
    call-checked against the prelude + per-unit generated defs.

    Nothing launches: kernel construction is pure AST/codegen work; the
    factory grids only rebuild @cache_kernel-cached Kernel records.
    """
    import importlib
    import inspect

    from warp_metal.frontend.kernels import Kernel
    from warp_metal.msl.prelude import PRELUDE

    # the prelude compiles into EVERY unit — its own internal delegation
    # must be checked first (this is the net that catches a family calling
    # a scalar core that was never defined)
    self_findings = self_check_prelude()
    self_errs = [f for f in self_findings if f.severity == "ERROR"]
    for f in self_findings:
        if f.severity == "ERROR":
            print(f"ERROR [prelude-self] {f.message}")
    total_err = len(self_errs)

    proto_db = parse_prototypes(PRELUDE)
    ok = fail = skipped = 0
    for modname in module_names:
        try:
            mod = importlib.import_module(modname)
        except Exception as e:  # noqa: BLE113 - report and continue
            print(f"MOD_FAIL [{modname}] {type(e).__name__}: {str(e)[:160]}")
            continue
        # module-level kernels
        for attr in sorted(dir(mod)):
            obj = getattr(mod, attr)
            if isinstance(obj, Kernel):
                status, errs = _check_one(obj, f"{modname}:{attr}", proto_db)
                total_err += errs
                ok += status == "ok"
                fail += status != "ok"
        # kernel-building factories: function defs whose source declares a
        # @wp.kernel (cache_kernel wrappers keep __wrapped__)
        checked_factories = set()
        for attr in sorted(dir(mod)):
            func = getattr(mod, attr)
            if not (inspect.isfunction(func) and hasattr(func, "__wrapped__")):
                continue
            try:
                src = inspect.getsource(getattr(func, "__wrapped__", func))
            except (OSError, TypeError):
                continue
            if not _RE_KERNEL_DECOR.search(src):
                continue
            key = id(func)
            if key in checked_factories:
                continue
            checked_factories.add(key)
            for args in _factory_grid(func):
                try:
                    kern = func(*args)
                except Exception:  # noqa: BLE113 - incompatible combo
                    skipped += 1
                    continue
                if not isinstance(kern, Kernel):
                    continue
                label = f"{modname}:{attr}{args if args else ''}"
                status, errs = _check_one(kern, label, proto_db)
                total_err += errs
                ok += status == "ok"
                fail += status != "ok"
    print(f"\nsweep: {ok} ok, {fail} with findings/gaps, {skipped} factory combos skipped, {total_err} errors")
    return total_err


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kernel", action="append", default=[], help="module:attr kernel spec (repeatable)")
    ap.add_argument("--scenario", choices=_TRACE_SCENARIOS, help="phase A: real-warp CPU closure trace (ground truth)")
    ap.add_argument(
        "--sweep",
        action="append",
        default=[],
        help="phase B: module sweep under the shim (module:attr specs or plain module names)",
    )
    ap.add_argument(
        "--prelude-self",
        action="store_true",
        help="check the prelude's own internal call delegation against its prototype DB",
    )
    args = ap.parse_args(argv)
    if args.prelude_self:
        findings = self_check_prelude()
        errs = [f for f in findings if f.severity == "ERROR"]
        for f in findings:
            print(f"{f.severity} {f.message}")
        print(f"prelude self-check: {len(errs)} errors, {len(findings) - len(errs)} warnings")
        return len(errs)
    if args.scenario:
        return _trace_scenario_closure(args.scenario)
    if args.sweep:
        names = []
        for spec in args.sweep:
            names.extend(s.strip() for s in spec.split(",") if s.strip())
        return run_sweep_mode(names)
    if args.kernel:
        return run_kernel_mode(args.kernel)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
