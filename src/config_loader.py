"""
config_loader.py — Proyecto GREEN-IA

Carga y valida config.yaml.
Centraliza las especificaciones estaticas de hardware para los dos
perfiles soportados por el experimento.

Uso:
    from src.config_loader import load_config, ExperimentConfig
    cfg = load_config()                      # carga config.yaml del root
    cfg = load_config(Path("otro.yaml"))     # ruta alternativa
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
CONFIG_FILE = ROOT / "config.yaml"

VALID_PROFILES = frozenset({"mac_m4", "windows_nvidia"})
VALID_DEVICES  = frozenset({"cpu", "gpu"})

# Especificaciones estaticas de cada perfil de hardware.
# Campos dinamicos (RAM detectada en runtime, version de Python, etc.) se
# obtienen por auto-deteccion en hardware_profile.detect_hardware().
HARDWARE_SPECS: dict[str, dict[str, Any]] = {
    "mac_m4": {
        # identificacion
        "machine_id"            : "apple_m4_mac_mini",
        "expected_os"           : "darwin",
        # CPU — Apple M4 10-core (4 Performance + 6 Efficiency)
        "cpu_model"             : "Apple M4 10-core CPU",
        "cpu_cores_physical"    : 10,
        "cpu_cores_logical"     : 10,  # no SMT en Apple Silicon
        "cpu_vendor"            : "apple_silicon",
        # RAM — memoria unificada (compartida CPU+GPU)
        "ram_total_gb"          : 16.0,
        "ram_type"              : "unified_memory",
        # GPU — Apple M4 10-core, backend Metal
        "gpu_model"             : "Apple M4 10-core GPU (Metal)",
        "gpu_backend"           : "metal",
        "gpu_vram_gb"           : 16.0,  # unified memory — no hay VRAM dedicada
        # threads y batch recomendados para llama.cpp
        "n_threads_recommended" : 8,
        "n_batch"               : 512,
    },
    "windows_nvidia": {
        # identificacion
        "machine_id"            : "windows_rtx4060",
        "expected_os"           : "windows",
        # CPU — Intel Core i7-13700H (6 P-cores + 8 E-cores)
        "cpu_model"             : "13th Gen Intel Core i7-13700H",
        "cpu_cores_physical"    : 14,   # 6 P-cores + 8 E-cores
        "cpu_cores_logical"     : 20,   # 6×2 (HT en P-cores) + 8×1 (sin HT en E-cores)
        "cpu_vendor"            : "intel",
        # RAM
        "ram_total_gb"          : 32.0,
        "ram_type"              : "ddr5",
        # GPU — NVIDIA RTX 4060 Laptop, VRAM dedicada 8 GB, backend CUDA
        "gpu_model"             : "NVIDIA GeForce RTX 4060 Laptop GPU",
        "gpu_backend"           : "cuda",
        "gpu_vram_gb"           : 8.0,
        "n_threads_recommended" : 14,   # todos los nucleos fisicos
        "n_batch"               : 512,
    },
}


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Configuracion leida de config.yaml.

    Inmutable (frozen=True) para garantizar consistencia durante el
    experimento — no se modifica despues de cargar.

    Campos con defaults: todos los campos nuevos tienen defaults que
    reproducen el comportamiento previo cuando el campo no esta en
    config.yaml, garantizando compatibilidad hacia atras.
    """
    # ── requeridos ─────────────────────────────────────────────────────────────
    hardware_profile: str   # mac_m4 | windows_nvidia
    execution_device: str   # cpu | gpu

    # ── power state (runtime_control block en config.yaml) ─────────────────────
    prevent_sleep: bool = True
    use_caffeinate_on_macos: bool = True  # activo solo si prevent_sleep = True

    # ── estabilizacion termica ─────────────────────────────────────────────────
    pause_before_start_seconds: int = 30
    pause_after_model_load_seconds: int = 20
    pause_after_experiment_seconds: int = 10
    cooldown_seconds_between_runs: int = 10
    cooldown_seconds_between_model_configs: int = 60

    # ── warmup ─────────────────────────────────────────────────────────────────
    warmup_runs_per_configuration: int = 1
    discard_warmup_runs: bool = True

    # ── calibracion de baseline (energy_calibration block en config.yaml) ────────
    energy_calibration_enabled: bool = True
    baseline_idle_seconds: int = 60
    baseline_repetitions: int = 3
    subtract_idle_baseline: bool = True
    save_baseline_runs: bool = True

    # ── analisis (analysis block en config.yaml) ───────────────────────────────
    # Metrica primaria para graficas y tablas de analisis.
    # Valores validos: "measured_energy" | "baseline_corrected_energy"
    primary_energy_metric: str = "baseline_corrected_energy"

    # ── llama.cpp params (llama_cpp_params block en config.yaml) ─────────────────
    n_ctx: int = 4096
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    echo: bool = False      # false: solo se cuentan los tokens de completion
    vary_seed_by_repetition: bool = False

    # ── context policy (context_policy block en config.yaml) ──────────────────
    on_context_overflow: str = "skip"       # "skip" | "warn"
    allow_prompt_truncation: bool = False

    # ── generation mode (generation_mode block en config.yaml) ────────────────
    # "deterministic_energy" — para la comparacion energetica principal
    # "stochastic_quality_variability" — modo secundario opcional
    generation_mode_name: str = "deterministic_energy"

    # ── metodos derivados ──────────────────────────────────────────────────────

    def n_gpu_layers(self) -> int:
        """
        Numero de capas a cargar en GPU para llama.cpp.

        Reglas:
          cpu -> 0   (inferencia solo en CPU)
          gpu -> -1  (todas las capas en GPU: Metal en mac_m4, CUDA en windows_nvidia)
        """
        return -1 if self.execution_device == "gpu" else 0

    def expected_backend(self) -> str:
        """Backend GPU esperado segun perfil + dispositivo de ejecucion."""
        if self.execution_device == "cpu":
            return "none"
        return HARDWARE_SPECS[self.hardware_profile]["gpu_backend"]

    def spec(self) -> dict[str, Any]:
        """Devuelve el spec estatico del perfil declarado."""
        return HARDWARE_SPECS[self.hardware_profile]

    def n_batch(self) -> int:
        """Tamano de batch recomendado segun el perfil de hardware."""
        return HARDWARE_SPECS[self.hardware_profile]["n_batch"]

    def n_threads(self) -> int:
        """Numero de threads recomendado segun el perfil de hardware."""
        return HARDWARE_SPECS[self.hardware_profile]["n_threads_recommended"]

    def effective_seed(self, repetition: int = 1) -> int:
        """
        Semilla efectiva para la repeticion indicada.

        Si vary_seed_by_repetition = False (default): siempre retorna self.seed.
        Si vary_seed_by_repetition = True: retorna self.seed + repetition - 1.

        Con temperature = 0.0 la generacion ya es determinista; este metodo
        permite documentar explicitamente que la semilla no varia.
        """
        if self.vary_seed_by_repetition:
            return self.seed + repetition - 1
        return self.seed

    def should_caffeinate(self) -> bool:
        """
        True solo si se debe activar caffeinate en este entorno.

        Condicion: prevent_sleep = True AND use_caffeinate_on_macos = True
                   AND hardware_profile = mac_m4.
        caffeinate es una utilidad macOS; no se usa en Windows.
        """
        return (
            self.prevent_sleep
            and self.use_caffeinate_on_macos
            and self.hardware_profile == "mac_m4"
        )


def load_config(path: Path = CONFIG_FILE) -> ExperimentConfig:
    """
    Carga y valida config.yaml.

    Todos los campos nuevos son opcionales con defaults razonables para
    compatibilidad hacia atras con config.yaml sin esos campos.

    Termina con sys.exit(1) si:
    - El archivo no existe (guia al usuario a crearlo desde la plantilla)
    - hardware_profile no es un valor valido
    - execution_device no es un valor valido
    """
    try:
        import yaml
    except ImportError:
        print("  ERROR: PyYAML no instalado. Ejecutar: pip install pyyaml")
        sys.exit(1)

    if not path.exists():
        print(
            f"\n  ERROR: {path} no encontrado."
            f"\n  Crear desde la plantilla:"
            f"\n    cp config.yaml.example config.yaml"
            f"\n  Luego editar hardware_profile y execution_device."
        )
        sys.exit(1)

    with path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    hw_profile  = str(raw.get("hardware_profile", "")).strip()
    exec_device = str(raw.get("execution_device", "")).strip()

    errors: list[str] = []
    if hw_profile not in VALID_PROFILES:
        errors.append(
            f"  hardware_profile='{hw_profile}' no valido."
            f" Opciones: {sorted(VALID_PROFILES)}"
        )
    if exec_device not in VALID_DEVICES:
        errors.append(
            f"  execution_device='{exec_device}' no valido."
            f" Opciones: {sorted(VALID_DEVICES)}"
        )
    if errors:
        print(f"\n  ERROR en {path.name}:")
        for e in errors:
            print(e)
        sys.exit(1)

    # bloques anidados — con fallback al nivel raiz para compatibilidad hacia atras
    rc: dict = raw.get("runtime_control")    or {}
    ec: dict = raw.get("energy_calibration") or {}
    an: dict = raw.get("analysis")           or {}
    lp: dict = raw.get("llama_cpp_params")   or {}
    cp: dict = raw.get("context_policy")     or {}
    gm: dict = raw.get("generation_mode")    or {}

    def _int(key: str, default: int, src: dict = raw) -> int:
        val = src.get(key, default)
        try:
            return max(0, int(val))
        except (TypeError, ValueError):
            return default

    def _float(key: str, default: float, src: dict = raw) -> float:
        val = src.get(key, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _bool(key: str, default: bool, src: dict = raw) -> bool:
        val = src.get(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "yes", "1")
        return bool(val)

    def _rc_int(key: str, default: int) -> int:
        return _int(key, _int(key, default), rc)

    def _rc_bool(key: str, default: bool) -> bool:
        return _bool(key, _bool(key, default), rc)

    def _ec_int(key: str, default: int) -> int:
        return _int(key, _int(key, default), ec)

    def _ec_bool(key: str, default: bool) -> bool:
        return _bool(key, _bool(key, default), ec)

    def _an_str(key: str, default: str) -> str:
        val = an.get(key, raw.get(key, default))
        return str(val).strip() if val is not None else default

    _valid_primary = frozenset({"measured_energy", "baseline_corrected_energy"})

    def _primary_metric() -> str:
        v = _an_str("primary_energy_metric", "baseline_corrected_energy")
        return v if v in _valid_primary else "baseline_corrected_energy"

    # Helpers para bloques llama_cpp_params / context_policy / generation_mode.
    # Prioridad: bloque especifico > nivel raiz > default.
    def _lp_int(key: str, default: int) -> int:
        return _int(key, _int(key, default), lp)

    def _lp_float(key: str, default: float) -> float:
        return _float(key, _float(key, default), lp)

    def _lp_bool(key: str, default: bool) -> bool:
        return _bool(key, _bool(key, default), lp)

    def _cp_str(key: str, default: str) -> str:
        val = cp.get(key, raw.get(key, default))
        return str(val).strip() if val is not None else default

    def _cp_bool(key: str, default: bool) -> bool:
        return _bool(key, _bool(key, default), cp)

    def _gm_bool(key: str, default: bool) -> bool:
        return _bool(key, _bool(key, default), gm)

    def _gm_str(key: str, default: str) -> str:
        val = gm.get(key, raw.get(key, default))
        return str(val).strip() if val is not None else default

    _gm_name = _gm_str("name", "deterministic_energy")

    # En stochastic_quality_variability, el bloque puede sobreescribir temperature.
    if _gm_name == "stochastic_quality_variability" and "temperature" in gm:
        _temperature = float(gm["temperature"])
    else:
        _temperature = _lp_float("temperature", 0.0)

    _valid_overflow = frozenset({"skip", "warn"})

    def _on_overflow() -> str:
        v = _cp_str("on_context_overflow", "skip")
        return v if v in _valid_overflow else "skip"

    return ExperimentConfig(
        hardware_profile                       = hw_profile,
        execution_device                       = exec_device,
        prevent_sleep                          = _rc_bool("prevent_sleep",                          True),
        use_caffeinate_on_macos                = _rc_bool("use_caffeinate_on_macos",                True),
        pause_before_start_seconds             = _rc_int ("pause_before_start_seconds",             30),
        pause_after_model_load_seconds         = _rc_int ("pause_after_model_load_seconds",         20),
        pause_after_experiment_seconds         = _rc_int ("pause_after_experiment_seconds",         10),
        cooldown_seconds_between_runs          = _rc_int ("cooldown_seconds_between_runs",          10),
        cooldown_seconds_between_model_configs = _rc_int ("cooldown_seconds_between_model_configs", 60),
        warmup_runs_per_configuration          = max(1, _rc_int("warmup_runs_per_configuration",    1)),
        discard_warmup_runs                    = _rc_bool("discard_warmup_runs",                    True),
        energy_calibration_enabled             = _ec_bool("enabled",                               True),
        baseline_idle_seconds                  = max(10, _ec_int("baseline_idle_seconds",           60)),
        baseline_repetitions                   = max(1,  _ec_int("baseline_repetitions",            3)),
        subtract_idle_baseline                 = _ec_bool("subtract_idle_baseline",                True),
        save_baseline_runs                     = _ec_bool("save_baseline_runs",                    True),
        primary_energy_metric                  = _primary_metric(),
        n_ctx                                  = max(512, _lp_int  ("n_ctx",                       4096)),
        max_tokens                             = max(64,  _lp_int  ("max_tokens",                   1024)),
        temperature                            = _temperature,
        top_p                                  = _lp_float("top_p",                                 1.0),
        seed                                   = _lp_int  ("seed",                                  42),
        echo                                   = _lp_bool ("echo",                                  False),
        vary_seed_by_repetition                = _gm_bool ("vary_seed_by_repetition",               False),
        on_context_overflow                    = _on_overflow(),
        allow_prompt_truncation                = _cp_bool ("allow_prompt_truncation",               False),
        generation_mode_name                   = _gm_name,
    )
