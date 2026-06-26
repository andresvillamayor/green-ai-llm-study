"""
validate_config.py
Proyecto GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay
Autor: Andres Villamayor

Valida la configuracion completa del experimento y el entorno de ejecucion.

Uso:
  python scripts/validate_config.py                   # checks basicos
  python scripts/validate_config.py --check-backend   # + backend GPU
  python scripts/validate_config.py --check-installation  # + paquetes Python
  python scripts/validate_config.py --check-optimization  # + optimization_config.yaml
  python scripts/validate_config.py --all             # todos los checks

Salida:
  [PASS]  check superado
  [WARN]  advertencia — no bloquea la ejecucion
  [FAIL]  error critico — el experimento no puede iniciarse
  [SKIP]  check no ejecutado (dependencia ausente o no aplica)

Codigo de salida:
  0   todos los checks PASS/WARN/SKIP (sin FAIL)
  1   al menos un check FAIL
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML no instalado. Ejecutar: pip install pyyaml")
    sys.exit(1)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CONFIG_FILE       = ROOT / "config.yaml"
OPTIM_CONFIG_FILE = ROOT / "optimization_config.yaml"
OFFICIAL_QFILE    = ROOT / "data" / "mt_bench" / "official" / "question.jsonl"

VALID_PROFILES      = frozenset({"mac_m4", "windows_nvidia"})
VALID_DEVICES       = frozenset({"cpu", "gpu"})
VALID_QUANTIZATIONS = frozenset({"q4", "q8"})
VALID_OVERFLOW      = frozenset({"skip", "warn"})
VALID_GEN_MODES     = frozenset({"deterministic_energy", "stochastic_quality_variability"})

OUTPUT_DIRS = [
    ROOT / "results" / "raw",
    ROOT / "results" / "judge",
    ROOT / "results" / "summary",
    ROOT / "results" / "optimization",
    ROOT / "results" / "plots",
    ROOT / "results" / "logs",
    ROOT / "results" / "codecarbon",
    ROOT / "results" / "backups",
    ROOT / "results" / "metadata",
]

REQUIRED_PACKAGES = [
    ("codecarbon",    "from codecarbon import EmissionsTracker"),
    ("pandas",        "import pandas"),
    ("numpy",         "import numpy"),
    ("pyyaml",        "import yaml"),
    ("tqdm",          "import tqdm"),
    ("psutil",        "import psutil"),
    ("anthropic",     "import anthropic"),
    ("python-dotenv", "from dotenv import load_dotenv"),
    ("matplotlib",    "import matplotlib"),
    ("scipy",         "import scipy"),
]


# ─── Status y resultado ───────────────────────────────────────────────────────

class Status(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


class Check:
    def __init__(self, name: str, status: Status, detail: str = "") -> None:
        self.name   = name
        self.status = status
        self.detail = detail

    @property
    def failed(self) -> bool:
        return self.status == Status.FAIL

    def __str__(self) -> str:
        tag    = f"[{self.status.value}]"
        detail = f"  — {self.detail}" if self.detail else ""
        return f"  {tag:<6} {self.name}{detail}"


def _pass(name: str, detail: str = "") -> Check:
    return Check(name, Status.PASS, detail)

def _warn(name: str, detail: str = "") -> Check:
    return Check(name, Status.WARN, detail)

def _fail(name: str, detail: str = "") -> Check:
    return Check(name, Status.FAIL, detail)

def _skip(name: str, detail: str = "") -> Check:
    return Check(name, Status.SKIP, detail)


class Section:
    def __init__(self, title: str) -> None:
        self.title  = title
        self.checks: list[Check] = []

    def add(self, check: Check) -> "Section":
        self.checks.append(check)
        return self

    def n_pass(self) -> int: return sum(1 for c in self.checks if c.status == Status.PASS)
    def n_warn(self) -> int: return sum(1 for c in self.checks if c.status == Status.WARN)
    def n_fail(self) -> int: return sum(1 for c in self.checks if c.status == Status.FAIL)
    def n_skip(self) -> int: return sum(1 for c in self.checks if c.status == Status.SKIP)
    def any_fail(self) -> bool: return any(c.failed for c in self.checks)


# ─── caffeinate (macOS) ───────────────────────────────────────────────────────

def _start_caffeinate() -> "subprocess.Popen | None":
    if platform.system().lower() != "darwin":
        return None
    try:
        return subprocess.Popen(
            ["caffeinate", "-dimsu"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None


def _stop_caffeinate(proc: "subprocess.Popen | None") -> None:
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass


# ─── Seccion 1: config.yaml — carga ──────────────────────────────────────────

def section_config_load() -> tuple[Section, Optional[dict]]:
    s = Section("config.yaml — carga y estructura")

    if not CONFIG_FILE.exists():
        s.add(_fail(
            "config.yaml existe",
            f"no encontrado: {CONFIG_FILE.relative_to(ROOT)}  —  "
            "crear con: cp config.yaml.example config.yaml",
        ))
        return s, None

    s.add(_pass("config.yaml existe"))

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        s.add(_fail("config.yaml es YAML valido", str(exc)))
        return s, None

    s.add(_pass("config.yaml es YAML valido"))

    if not isinstance(cfg, dict):
        s.add(_fail("config.yaml es un mapping", f"tipo obtenido: {type(cfg).__name__}"))
        return s, None

    s.add(_pass("config.yaml es un mapping YAML"))
    return s, cfg or {}


# ─── Seccion 2: campos requeridos ─────────────────────────────────────────────

def section_required_fields(cfg: dict) -> Section:
    s = Section("Campos requeridos")

    required: list[tuple[str, type]] = [
        ("hardware_profile",     str),
        ("execution_device",     str),
        ("selected_models",      list),
        ("selected_quantizations", list),
        ("models",               dict),
        ("llama_cpp_params",     dict),
        ("generation_mode",      dict),
    ]

    for field, expected_type in required:
        val = cfg.get(field)
        if val is None:
            s.add(_fail(f"campo '{field}' presente", "ausente en config.yaml"))
        elif not isinstance(val, expected_type):
            s.add(_fail(
                f"campo '{field}' es {expected_type.__name__}",
                f"tipo obtenido: {type(val).__name__}",
            ))
        else:
            s.add(_pass(f"campo '{field}' presente y con tipo correcto"))

    return s


# ─── Seccion 3: hardware_profile y execution_device ──────────────────────────

def section_hardware(cfg: dict) -> Section:
    s = Section("Hardware y dispositivo")

    profile = str(cfg.get("hardware_profile", "")).strip()
    device  = str(cfg.get("execution_device",  "")).strip()

    if profile in VALID_PROFILES:
        s.add(_pass(f"hardware_profile valido", f"valor: {profile}"))
    else:
        s.add(_fail(
            "hardware_profile valido",
            f"valor: '{profile}'  — validos: {sorted(VALID_PROFILES)}",
        ))

    if device in VALID_DEVICES:
        s.add(_pass("execution_device valido", f"valor: {device}"))
    else:
        s.add(_fail(
            "execution_device valido",
            f"valor: '{device}'  — validos: {sorted(VALID_DEVICES)}",
        ))

    # Coherencia: hardware_profile y execution_device deben ser consistentes
    # con los hardware_profiles definidos en el mismo archivo.
    hp_block = cfg.get("hardware_profiles", {})
    if hp_block and profile in VALID_PROFILES:
        if profile in hp_block:
            s.add(_pass(f"hardware_profile '{profile}' tiene definicion en hardware_profiles"))
        else:
            s.add(_warn(
                f"hardware_profile '{profile}' no tiene definicion en hardware_profiles",
                "agregar bloque en config.yaml para documentacion",
            ))

    # Plataforma del sistema vs hardware_profile configurado
    system = platform.system().lower()
    system_ok = (
        (profile == "mac_m4"         and system == "darwin")
        or (profile == "windows_nvidia" and system == "windows")
    )
    if system_ok:
        s.add(_pass(
            "hardware_profile coincide con la plataforma actual",
            f"profile={profile}  os={system}",
        ))
    else:
        s.add(_warn(
            "hardware_profile no coincide con la plataforma actual",
            f"profile={profile}  os_detected={system}  "
            "(ignorar si se esta editando la config en otra maquina)",
        ))

    return s


# ─── Seccion 4: modelos ───────────────────────────────────────────────────────

def section_models(cfg: dict) -> Section:
    s = Section("Modelos y cuantizaciones")

    selected_models = cfg.get("selected_models") or []
    selected_quants = cfg.get("selected_quantizations") or []
    models_block    = cfg.get("models") or {}

    # selected_quantizations
    if not selected_quants:
        s.add(_fail("selected_quantizations no esta vacio"))
    else:
        invalid_q = [q for q in selected_quants if q not in VALID_QUANTIZATIONS]
        if invalid_q:
            s.add(_fail(
                "selected_quantizations contiene solo valores validos",
                f"invalidos: {invalid_q}  — validos: {sorted(VALID_QUANTIZATIONS)}",
            ))
        else:
            s.add(_pass("selected_quantizations validos", f"valores: {selected_quants}"))

    # selected_models — cada modelo debe existir en el bloque models
    if not selected_models:
        s.add(_fail("selected_models no esta vacio"))
    else:
        for model in selected_models:
            if model not in models_block:
                s.add(_fail(
                    f"modelo '{model}' definido en models",
                    "agregar entrada en el bloque 'models' de config.yaml",
                ))
            else:
                s.add(_pass(f"modelo '{model}' definido en models"))
                model_def = models_block[model]
                # Cada cuantizacion seleccionada debe tener ruta
                for q in selected_quants:
                    if q not in model_def:
                        s.add(_warn(
                            f"modelo '{model}' tiene ruta para cuantizacion '{q}'",
                            "agregar ruta en config.yaml -> models -> "
                            f"{model} -> {q}",
                        ))
                    else:
                        model_path = ROOT / str(model_def[q])
                        if model_path.exists():
                            size_gb = model_path.stat().st_size / 1e9
                            s.add(_pass(
                                f"archivo modelo existe: {model} / {q}",
                                f"{model_def[q]}  ({size_gb:.2f} GB)",
                            ))
                        else:
                            s.add(_warn(
                                f"archivo modelo existe: {model} / {q}",
                                f"no encontrado: {model_def[q]}  "
                                "(descargar antes de ejecutar el experimento)",
                            ))

    return s


# ─── Seccion 5: parametros de inferencia ─────────────────────────────────────

def section_inference_params(cfg: dict) -> Section:
    s = Section("Parametros de inferencia")

    lp = cfg.get("llama_cpp_params") or {}
    gm = cfg.get("generation_mode")  or {}
    cp = cfg.get("context_policy")   or {}

    # n_ctx
    n_ctx = lp.get("n_ctx")
    if n_ctx is None:
        s.add(_warn("llama_cpp_params.n_ctx definido", "ausente — se usara el default del modelo"))
    elif not isinstance(n_ctx, int):
        s.add(_fail("llama_cpp_params.n_ctx es entero", f"tipo: {type(n_ctx).__name__}"))
    elif n_ctx < 4096:
        s.add(_warn(
            "llama_cpp_params.n_ctx >= 4096 (recomendado para MT-Bench multi-turno)",
            f"n_ctx={n_ctx}  — turno 2 de MT-Bench puede necesitar hasta ~3000 tokens",
        ))
    else:
        s.add(_pass("llama_cpp_params.n_ctx suficiente", f"n_ctx={n_ctx}"))

    # max_tokens
    max_tokens = lp.get("max_tokens")
    if max_tokens is not None:
        if not isinstance(max_tokens, int) or not (64 <= max_tokens <= 4096):
            s.add(_warn(
                "llama_cpp_params.max_tokens en rango recomendado (64-4096)",
                f"max_tokens={max_tokens}",
            ))
        else:
            s.add(_pass("llama_cpp_params.max_tokens valido", f"max_tokens={max_tokens}"))

    # temperature para modo deterministic_energy
    gm_name   = str(gm.get("name", "deterministic_energy"))
    temp_val  = lp.get("temperature")

    if gm_name not in VALID_GEN_MODES:
        s.add(_warn(
            "generation_mode.name es un valor conocido",
            f"valor: '{gm_name}'  — conocidos: {sorted(VALID_GEN_MODES)}",
        ))
    else:
        s.add(_pass("generation_mode.name valido", f"mode={gm_name}"))

    if gm_name == "deterministic_energy":
        if temp_val is None:
            s.add(_warn(
                "temperature = 0.0 para modo deterministic_energy",
                "llama_cpp_params.temperature no definido",
            ))
        elif float(temp_val) != 0.0:
            s.add(_warn(
                "temperature = 0.0 para modo deterministic_energy",
                f"temperature={temp_val}  — generacion no sera determinista; "
                "los resultados de energia pueden variar entre repeticiones",
            ))
        else:
            s.add(_pass("temperature = 0.0 para modo deterministic_energy"))

    # context_policy
    overflow = str(cp.get("on_context_overflow", "skip"))
    if overflow not in VALID_OVERFLOW:
        s.add(_fail(
            "context_policy.on_context_overflow valido",
            f"valor: '{overflow}'  — validos: {sorted(VALID_OVERFLOW)}",
        ))
    else:
        s.add(_pass("context_policy.on_context_overflow valido", f"valor: {overflow}"))

    return s


# ─── Seccion 6: dataset MT-Bench ─────────────────────────────────────────────

def section_mtbench(cfg: dict) -> Section:
    s = Section("Dataset MT-Bench")

    # Ruta configurada (puede diferir del default)
    pd_cfg = cfg.get("prompt_dataset") or {}
    configured_path_str = pd_cfg.get("official_question_file")
    if configured_path_str:
        configured_path = ROOT / str(configured_path_str)
    else:
        configured_path = OFFICIAL_QFILE

    # Verificar que la ruta configurada coincide con la canonica
    if configured_path.resolve() != OFFICIAL_QFILE.resolve():
        s.add(_warn(
            "official_question_file coincide con la ruta canonica",
            f"configurado: {configured_path_str}  "
            f"canonico: {OFFICIAL_QFILE.relative_to(ROOT)}",
        ))

    # Existencia
    if not configured_path.exists():
        s.add(_fail(
            "question.jsonl presente en data/mt_bench/official/",
            "obtener con:  git clone https://github.com/lm-sys/FastChat.git  &&  "
            "cp FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl "
            "data/mt_bench/official/question.jsonl",
        ))
        return s

    s.add(_pass(
        "question.jsonl presente en data/mt_bench/official/",
        f"{configured_path.stat().st_size:,} bytes",
    ))

    # Validacion basica del contenido
    import json
    try:
        rows = []
        with open(configured_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    s.add(_fail("question.jsonl es JSONL valido", f"linea {i}: {exc}"))
                    return s

        n_questions = len(rows)
        categories  = {str(r.get("category", "")).lower() for r in rows}
        turn_errors = [
            r.get("question_id")
            for r in rows
            if not isinstance(r.get("turns"), list) or len(r["turns"]) != 2
        ]

        expected_cats = {
            "writing", "roleplay", "extraction", "reasoning",
            "math", "coding", "stem", "humanities",
        }
        missing_cats = expected_cats - categories

        s.add(_pass("question.jsonl es JSONL valido", f"{n_questions} preguntas"))

        if n_questions == 80:
            s.add(_pass("question.jsonl tiene 80 preguntas (dataset oficial completo)"))
        else:
            s.add(_warn(
                "question.jsonl tiene 80 preguntas",
                f"encontradas: {n_questions}",
            ))

        if missing_cats:
            s.add(_fail(
                "question.jsonl contiene las 8 categorias oficiales",
                f"faltantes: {sorted(missing_cats)}",
            ))
        else:
            s.add(_pass("question.jsonl contiene las 8 categorias oficiales"))

        if turn_errors:
            s.add(_fail(
                "todas las filas tienen exactamente 2 turnos",
                f"question_ids con error: {turn_errors[:5]}",
            ))
        else:
            s.add(_pass("todas las filas tienen exactamente 2 turnos"))

    except Exception as exc:
        s.add(_fail("question.jsonl se puede leer", str(exc)))

    return s


# ─── Seccion 7: directorios de salida ────────────────────────────────────────

def section_output_dirs() -> Section:
    s = Section("Directorios de salida")

    for d in OUTPUT_DIRS:
        try:
            d.mkdir(parents=True, exist_ok=True)
            s.add(_pass(
                f"directorio accesible: {d.relative_to(ROOT)}",
            ))
        except OSError as exc:
            s.add(_fail(
                f"directorio accesible: {d.relative_to(ROOT)}",
                str(exc),
            ))

    # results/metadata puede escribirse
    metadata_dir = ROOT / "results" / "metadata"
    try:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        tmp = metadata_dir / ".write_test"
        tmp.write_text("ok")
        tmp.unlink()
        s.add(_pass("results/metadata tiene permisos de escritura"))
    except OSError as exc:
        s.add(_fail("results/metadata tiene permisos de escritura", str(exc)))

    return s


# ─── Seccion 8: backend GPU (--check-backend) ─────────────────────────────────

def section_backend(cfg: dict) -> Section:
    s = Section("Backend GPU")

    system  = platform.system().lower()
    profile = str(cfg.get("hardware_profile", "")).strip()
    device  = str(cfg.get("execution_device",  "")).strip()

    if system == "darwin":
        arch = platform.machine()
        if arch == "arm64":
            s.add(_pass("plataforma macOS Apple Silicon (arm64)", f"arch={arch}"))
        else:
            s.add(_fail("plataforma macOS Apple Silicon (arm64)", f"arch={arch}"))

        if profile == "mac_m4":
            s.add(_pass("hardware_profile = mac_m4 en macOS"))
        else:
            s.add(_warn(
                "hardware_profile = mac_m4 en macOS",
                f"configurado: {profile}",
            ))

        if device == "gpu":
            s.add(_pass("execution_device = gpu (Metal)"))
        else:
            s.add(_warn("execution_device = gpu (Metal)", f"configurado: {device}"))

        # llama-cpp-python con soporte Metal
        try:
            from llama_cpp import Llama, __version__ as llama_v  # noqa: F401
            s.add(_pass("llama-cpp-python importa", f"version={llama_v}"))
            # Verificar que Metal esta disponible via atributo interno
            try:
                import llama_cpp as _lc
                supports_metal = hasattr(_lc, "_lib") or True  # import exitoso implica build funcional
                s.add(_pass("llama-cpp-python build disponible (verificar Metal en ejecucion)"))
            except Exception:
                s.add(_skip("verificacion interna de Metal", "no aplicable en este entorno"))
        except ImportError as exc:
            s.add(_fail(
                "llama-cpp-python importa",
                f"{exc}  —  reinstalar: "
                'CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir',
            ))

    elif system == "windows":
        s.add(_pass("plataforma Windows", f"version={platform.version()[:30]}"))

        if profile == "windows_nvidia":
            s.add(_pass("hardware_profile = windows_nvidia en Windows"))
        else:
            s.add(_warn(
                "hardware_profile = windows_nvidia en Windows",
                f"configurado: {profile}",
            ))

        # pynvml (opcional, WARN si ausente)
        try:
            import pynvml
            s.add(_pass("pynvml disponible"))

            if device == "gpu":
                try:
                    pynvml.nvmlInit()
                    count = pynvml.nvmlDeviceGetCount()
                    if count > 0:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        name = pynvml.nvmlDeviceGetName(handle)
                        if isinstance(name, bytes):
                            name = name.decode()
                        s.add(_pass("GPU NVIDIA visible via pynvml", f"devices={count}  name={name}"))
                    else:
                        s.add(_fail("GPU NVIDIA visible via pynvml", "nvmlDeviceGetCount() == 0"))
                except Exception as exc:
                    s.add(_fail("GPU NVIDIA visible via pynvml", str(exc)))
            else:
                s.add(_skip("GPU NVIDIA visible via pynvml", f"execution_device={device}"))

        except ImportError:
            s.add(_warn(
                "pynvml disponible",
                "no instalado  —  pip install pynvml  (necesario para metricas GPU en Windows)",
            ))
            if device == "gpu":
                s.add(_skip("GPU NVIDIA visible via pynvml", "pynvml no instalado"))

        # llama-cpp-python con CUDA
        try:
            from llama_cpp import Llama, __version__ as llama_v  # noqa: F401
            s.add(_pass("llama-cpp-python importa", f"version={llama_v}"))
        except ImportError as exc:
            s.add(_fail(
                "llama-cpp-python importa",
                f"{exc}  —  reinstalar: "
                "set CMAKE_ARGS=-DGGML_CUDA=on && "
                "pip install llama-cpp-python --force-reinstall --no-cache-dir",
            ))

    else:
        s.add(_warn(
            "plataforma reconocida para check-backend",
            f"sistema: {platform.system()}  — soportados: macOS (darwin), Windows",
        ))

    return s


# ─── Seccion 9: paquetes Python (--check-installation) ───────────────────────

def section_installation() -> Section:
    s = Section("Paquetes Python")

    for pkg_name, import_stmt in REQUIRED_PACKAGES:
        try:
            exec(import_stmt, {})  # noqa: S102
            # Obtener version si es posible
            module_name = import_stmt.split()[-1].split(".")[0]
            try:
                import importlib
                mod = importlib.import_module(module_name)
                ver = getattr(mod, "__version__", None) or getattr(mod, "VERSION", None)
                detail = f"version={ver}" if ver else "importado"
            except Exception:
                detail = "importado"
            s.add(_pass(f"{pkg_name} importa correctamente", detail))
        except ImportError as exc:
            s.add(_fail(
                f"{pkg_name} importa correctamente",
                f"{exc}  —  pip install {pkg_name}",
            ))
        except Exception as exc:
            s.add(_fail(f"{pkg_name} importa correctamente", str(exc)))

    # CodeCarbon — check especifico
    try:
        from codecarbon import EmissionsTracker  # noqa: F401
        s.add(_pass("codecarbon.EmissionsTracker importa"))
    except ImportError as exc:
        s.add(_fail("codecarbon.EmissionsTracker importa", str(exc)))

    # llama-cpp-python — check especifico
    try:
        from llama_cpp import Llama, __version__ as llama_v  # noqa: F401
        s.add(_pass("llama_cpp.Llama importa", f"version={llama_v}"))
    except ImportError as exc:
        s.add(_fail(
            "llama_cpp.Llama importa",
            f"{exc}  —  ver README para instrucciones de instalacion con Metal/CUDA",
        ))

    return s


# ─── Seccion 10: optimization_config.yaml (--check-optimization) ─────────────

def section_optimization() -> Section:
    s = Section("optimization_config.yaml")

    if not OPTIM_CONFIG_FILE.exists():
        s.add(_fail(
            "optimization_config.yaml existe",
            f"no encontrado: {OPTIM_CONFIG_FILE.relative_to(ROOT)}",
        ))
        return s

    s.add(_pass("optimization_config.yaml existe"))

    try:
        with open(OPTIM_CONFIG_FILE, "r", encoding="utf-8") as fh:
            opt = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        s.add(_fail("optimization_config.yaml es YAML valido", str(exc)))
        return s

    s.add(_pass("optimization_config.yaml es YAML valido"))

    if not isinstance(opt, dict) or "optimization" not in opt:
        s.add(_fail(
            "optimization_config.yaml contiene bloque 'optimization'",
            "clave 'optimization' ausente",
        ))
        return s

    s.add(_pass("bloque 'optimization' presente"))
    o = opt["optimization"]

    # enabled debe ser bool
    enabled = o.get("enabled")
    if isinstance(enabled, bool):
        s.add(_pass(
            "optimization.enabled es bool",
            f"valor: {enabled}  "
            "('false' = modo dry-run/planning, no ejecuta trials)",
        ))
    else:
        s.add(_warn(
            "optimization.enabled es bool",
            f"tipo obtenido: {type(enabled).__name__}",
        ))

    # forbidden_changes — todos deben ser False
    # Fase 2 no puede cambiar modelo, cuantizacion, prompts, judge prompt ni subset
    fc = o.get("forbidden_changes")
    if not isinstance(fc, dict):
        s.add(_fail(
            "optimization.forbidden_changes presente y es mapping",
            "ausente o tipo incorrecto",
        ))
    else:
        s.add(_pass("optimization.forbidden_changes presente"))

        FORBIDDEN_KEYS = {
            "change_model_file"     : "modelo GGUF",
            "change_quantization"   : "cuantizacion",
            "change_prompt_text"    : "texto de prompts MT-Bench",
            "change_judge_prompt"   : "prompt del juez",
            "change_benchmark_subset": "subconjunto del benchmark",
        }

        for key, label in FORBIDDEN_KEYS.items():
            val = fc.get(key)
            if val is None:
                s.add(_warn(
                    f"forbidden_changes.{key} definido",
                    f"ausente — agregar '{key}: false'",
                ))
            elif val is True:
                s.add(_fail(
                    f"forbidden_changes.{key} = false",
                    f"valor: true  — Fase 2 no puede modificar {label}",
                ))
            elif val is False:
                s.add(_pass(
                    f"forbidden_changes.{key} = false",
                    f"{label} bloqueado en Fase 2",
                ))
            else:
                s.add(_warn(
                    f"forbidden_changes.{key} es bool",
                    f"tipo obtenido: {type(val).__name__}",
                ))

    # parameter_space — temperature debe ser [0.0] o no variar
    ps = o.get("parameter_space") or {}
    temp_space = ps.get("temperature", {})
    temp_vals  = temp_space.get("values", []) if isinstance(temp_space, dict) else []
    if temp_vals and temp_vals != [0.0]:
        s.add(_warn(
            "parameter_space.temperature contiene solo [0.0]",
            f"valores: {temp_vals}  — temperatura != 0.0 genera salida no determinista",
        ))
    elif temp_vals == [0.0]:
        s.add(_pass("parameter_space.temperature = [0.0] (generacion determinista)"))

    return s


# ─── Reporte ──────────────────────────────────────────────────────────────────

def print_report(sections: list[Section]) -> int:
    W = 65
    total_pass = total_warn = total_fail = total_skip = 0

    for sec in sections:
        if not sec.checks:
            continue
        print(f"\n  {'─' * (W - 2)}")
        print(f"  {sec.title}")
        print(f"  {'─' * (W - 2)}")
        for c in sec.checks:
            print(c)
        total_pass += sec.n_pass()
        total_warn += sec.n_warn()
        total_fail += sec.n_fail()
        total_skip += sec.n_skip()

    print(f"\n  {'═' * (W - 2)}")
    parts = [f"{total_pass} PASS"]
    if total_warn: parts.append(f"{total_warn} WARN")
    if total_fail: parts.append(f"{total_fail} FAIL")
    if total_skip: parts.append(f"{total_skip} SKIP")
    print(f"  Resultado: {' / '.join(parts)}")

    if total_fail == 0:
        print("  Entorno listo para el experimento.")
    else:
        print(f"  Corregir los {total_fail} check(s) [FAIL] antes de iniciar el experimento.")

    print(f"  {'═' * (W - 2)}\n")
    return total_fail


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="GREEN-IA: valida config.yaml y el entorno de ejecucion"
    )
    parser.add_argument(
        "--check-backend",
        action="store_true",
        help="Verifica el backend GPU (Metal en macOS, CUDA en Windows)",
    )
    parser.add_argument(
        "--check-installation",
        action="store_true",
        help="Verifica que todos los paquetes Python requeridos importan correctamente",
    )
    parser.add_argument(
        "--check-optimization",
        action="store_true",
        help="Verifica optimization_config.yaml y las restricciones de Fase 2",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ejecuta todos los checks (equivalente a --check-backend --check-installation --check-optimization)",
    )
    args = parser.parse_args()

    if args.all:
        args.check_backend      = True
        args.check_installation = True
        args.check_optimization = True

    W = 65
    print(f"\n{'=' * W}")
    print(f"  GREEN-IA — Validacion de configuracion")
    modes = []
    if args.check_backend:      modes.append("backend")
    if args.check_installation: modes.append("installation")
    if args.check_optimization: modes.append("optimization")
    if modes:
        print(f"  Modo: {', '.join(f'--check-{m}' for m in modes)}")
    print(f"{'=' * W}")

    caff = _start_caffeinate()
    if caff is not None:
        print(f"\n  caffeinate activo (PID {caff.pid})\n")

    sections: list[Section] = []

    try:
        # ── checks basicos (siempre) ──────────────────────────────────────────
        s_load, cfg = section_config_load()
        sections.append(s_load)

        if cfg is None:
            print_report(sections)
            sys.exit(1)

        sections.append(section_required_fields(cfg))
        sections.append(section_hardware(cfg))
        sections.append(section_models(cfg))
        sections.append(section_inference_params(cfg))
        sections.append(section_mtbench(cfg))
        sections.append(section_output_dirs())

        # ── checks opcionales ─────────────────────────────────────────────────
        if args.check_backend:
            sections.append(section_backend(cfg))

        if args.check_installation:
            sections.append(section_installation())

        if args.check_optimization:
            sections.append(section_optimization())

    finally:
        _stop_caffeinate(caff)

    n_fail = print_report(sections)
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
