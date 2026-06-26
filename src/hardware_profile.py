"""
hardware_profile.py
Proyecto GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay

Deteccion automatica de hardware para:
  - Apple M4 Mac Mini (backend Metal, Apple Silicon)
  - Windows + NVIDIA RTX 4060 (backend CUDA)

Justificacion metodologica:
  Comparar el mismo LLM en arquitecturas distintas (Apple Silicon vs
  NVIDIA CUDA) requiere documentar el entorno exacto para garantizar
  reproducibilidad y atribuir correctamente las diferencias observadas.
  Strubell et al. (2019). Energy and Policy Considerations for Deep
  Learning in NLP. ACL 2019. arXiv:1906.02629 — Tabla 1 documenta
  hardware como variable independiente del consumo.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional

import psutil


@dataclass
class HardwareProfile:
    """Perfil completo del hardware donde corre el experimento."""

    # Identificacion de maquina
    machine_id: str          # 'apple_m4_mac_mini' | 'windows_rtx4060' | 'other'
    os_platform: str         # 'darwin' | 'windows' | 'linux'
    os_version: str

    # CPU
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_vendor: str          # 'apple_silicon' | 'intel' | 'amd' | 'unknown'

    # RAM
    ram_total_gb: float
    ram_type: str            # 'unified_memory' (M4) | 'ddr4' | 'ddr5' | 'unknown'

    # GPU
    gpu_available: bool
    gpu_model: str
    gpu_backend: str         # 'metal' | 'cuda' | 'none'
    gpu_vram_gb: float       # 0 si no hay GPU discreta

    # Parametros llama.cpp derivados del hardware
    n_gpu_layers_for_gpu: int   # -1 para Metal/CUDA
    n_gpu_layers_for_cpu: int   # siempre 0
    n_threads_recommended: int

    # Emisiones
    country_iso: str
    carbon_intensity_g_kwh: int

    # Python y llama.cpp
    python_version: str
    llama_cpp_version: str

    def to_dict(self) -> dict:
        return asdict(self)

    def get_n_gpu_layers(self, device: str) -> int:
        if device == "gpu":
            return self.n_gpu_layers_for_gpu
        return self.n_gpu_layers_for_cpu


def _get_llama_version() -> str:
    try:
        from llama_cpp import __version__ as v
        return v
    except ImportError:
        return "no_instalado"


def _detect_nvidia_gpu() -> Optional[str]:
    """Retorna nombre de la GPU NVIDIA si nvidia-smi esta disponible."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.strip().split("\n")[0].strip()
    except Exception:
        return None


def _detect_nvidia_vram_gb() -> float:
    """Retorna VRAM en GB de la primera GPU NVIDIA."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        mib = float(out.strip().split("\n")[0].strip())
        return round(mib / 1024, 1)
    except Exception:
        return 0.0


def _get_mac_chip() -> str:
    """Retorna el nombre del chip en macOS via sysctl."""
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out if out else "Apple Silicon"
    except Exception:
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.model"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return out if out else "Apple Silicon"
        except Exception:
            return "Apple Silicon"


def detect_hardware(
    experiment_config=None,      # Optional[ExperimentConfig] — duck-typed para evitar import circular
    country_iso: str = "PRY",
    carbon_intensity: int = 26,
) -> HardwareProfile:
    """
    Construye un HardwareProfile para el experimento.

    Si se proporciona experiment_config (cargado de config.yaml), se usan
    las especificaciones declaradas del perfil (machine_id, gpu_backend,
    CPU, RAM, etc.) y se valida que el OS actual coincida con el esperado.

    Si experiment_config es None, se hace auto-deteccion del hardware
    (comportamiento original, usado como fallback).

    Parametros:
        experiment_config: ExperimentConfig de src.config_loader (o None)
        country_iso: codigo ISO del pais para CodeCarbon
        carbon_intensity: factor de emision en gCO2eq/kWh
    """
    python_ver = platform.python_version()
    llama_ver  = _get_llama_version()
    system     = platform.system().lower()

    # ─ Ruta 1: perfil declarado en config.yaml ────────────────────────────────
    if experiment_config is not None:
        spec        = experiment_config.spec()          # dict de HARDWARE_SPECS
        expected_os = spec["expected_os"]               # "darwin" | "windows"

        if system != expected_os:
            import sys as _sys
            print(
                f"\n  ERROR: hardware_profile='{experiment_config.hardware_profile}'"
                f" espera OS='{expected_os}', detectado OS='{system}'."
                f"\n  Verificar config.yaml y ejecutar en la maquina correcta."
            )
            _sys.exit(1)

        os_ver = platform.mac_ver()[0] if system == "darwin" else platform.version()

        return HardwareProfile(
            machine_id             = spec["machine_id"],
            os_platform            = expected_os,
            os_version             = os_ver,
            cpu_model              = spec["cpu_model"],
            cpu_cores_physical     = spec["cpu_cores_physical"],
            cpu_cores_logical      = spec["cpu_cores_logical"],
            cpu_vendor             = spec["cpu_vendor"],
            ram_total_gb           = spec["ram_total_gb"],
            ram_type               = spec["ram_type"],
            gpu_available          = True,              # ambos perfiles tienen GPU
            gpu_model              = spec["gpu_model"],
            gpu_backend            = spec["gpu_backend"],
            gpu_vram_gb            = spec["gpu_vram_gb"],
            n_gpu_layers_for_gpu   = -1,               # todas las capas en GPU
            n_gpu_layers_for_cpu   = 0,
            n_threads_recommended  = spec["n_threads_recommended"],
            country_iso            = country_iso,
            carbon_intensity_g_kwh = carbon_intensity,
            python_version         = python_ver,
            llama_cpp_version      = llama_ver,
        )

    # ─ Ruta 2: auto-deteccion (fallback cuando no hay config.yaml) ────────────
    system = platform.system().lower()
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    logical_cores = psutil.cpu_count(logical=True) or 4
    physical_cores = psutil.cpu_count(logical=False) or logical_cores
    python_ver = platform.python_version()
    llama_ver = _get_llama_version()
    os_ver = platform.version()

    if system == "darwin":
        chip = _get_mac_chip()
        is_apple_silicon = "Apple" in chip

        profile = HardwareProfile(
            machine_id="apple_m4_mac_mini" if "M4" in chip else "apple_silicon_mac",
            os_platform="darwin",
            os_version=platform.mac_ver()[0],
            cpu_model=chip,
            cpu_cores_physical=physical_cores,
            cpu_cores_logical=logical_cores,
            cpu_vendor="apple_silicon" if is_apple_silicon else "intel",
            ram_total_gb=ram_gb,
            ram_type="unified_memory" if is_apple_silicon else "lpddr4",
            gpu_available=is_apple_silicon,  # Metal siempre disponible
            gpu_model=f"{chip} GPU (Metal)" if is_apple_silicon else "none",
            gpu_backend="metal" if is_apple_silicon else "none",
            gpu_vram_gb=ram_gb if is_apple_silicon else 0.0,
            n_gpu_layers_for_gpu=-1 if is_apple_silicon else 0,
            n_gpu_layers_for_cpu=0,
            n_threads_recommended=min(physical_cores, 8),
            country_iso=country_iso,
            carbon_intensity_g_kwh=carbon_intensity,
            python_version=python_ver,
            llama_cpp_version=llama_ver,
        )

    elif system == "windows":
        nvidia_name = _detect_nvidia_gpu()
        nvidia_vram = _detect_nvidia_vram_gb() if nvidia_name else 0.0
        cpu_name = platform.processor() or "Intel Core"

        profile = HardwareProfile(
            machine_id="windows_rtx4060" if nvidia_name and "4060" in nvidia_name else "windows_nvidia",
            os_platform="windows",
            os_version=os_ver,
            cpu_model=cpu_name,
            cpu_cores_physical=physical_cores,
            cpu_cores_logical=logical_cores,
            cpu_vendor="intel" if "intel" in cpu_name.lower() else "amd",
            ram_total_gb=ram_gb,
            ram_type="ddr5",
            gpu_available=nvidia_name is not None,
            gpu_model=nvidia_name or "none",
            gpu_backend="cuda" if nvidia_name else "none",
            gpu_vram_gb=nvidia_vram,
            n_gpu_layers_for_gpu=-1 if nvidia_name else 0,
            n_gpu_layers_for_cpu=0,
            n_threads_recommended=min(physical_cores, 8),
            country_iso=country_iso,
            carbon_intensity_g_kwh=carbon_intensity,
            python_version=python_ver,
            llama_cpp_version=llama_ver,
        )

    else:
        # Linux u otro sistema
        nvidia_name = _detect_nvidia_gpu()
        nvidia_vram = _detect_nvidia_vram_gb() if nvidia_name else 0.0

        profile = HardwareProfile(
            machine_id="linux_other",
            os_platform=system,
            os_version=os_ver,
            cpu_model=platform.processor() or "unknown",
            cpu_cores_physical=physical_cores,
            cpu_cores_logical=logical_cores,
            cpu_vendor="unknown",
            ram_total_gb=ram_gb,
            ram_type="ddr4",
            gpu_available=nvidia_name is not None,
            gpu_model=nvidia_name or "none",
            gpu_backend="cuda" if nvidia_name else "none",
            gpu_vram_gb=nvidia_vram,
            n_gpu_layers_for_gpu=-1 if nvidia_name else 0,
            n_gpu_layers_for_cpu=0,
            n_threads_recommended=min(physical_cores, 8),
            country_iso=country_iso,
            carbon_intensity_g_kwh=carbon_intensity,
            python_version=python_ver,
            llama_cpp_version=llama_ver,
        )

    return profile


def print_hardware_profile(
    hw: HardwareProfile,
    experiment_config=None,   # Optional[ExperimentConfig] — para mostrar fuente y device
) -> None:
    src = f"config.yaml ({experiment_config.hardware_profile})" if experiment_config else "auto-deteccion"
    print(f"  fuente perfil : {src}")
    print(f"  maquina       : {hw.machine_id}")
    if experiment_config is not None:
        print(f"  dispositivo   : {experiment_config.execution_device}  "
              f"(n_gpu_layers = {hw.get_n_gpu_layers(experiment_config.execution_device)},"
              f" backend esperado = {experiment_config.expected_backend()})")
    print(f"  sistema       : {hw.os_platform} {hw.os_version}")
    print(f"  CPU           : {hw.cpu_model}")
    print(f"  nucleos       : {hw.cpu_cores_physical} fisicos / {hw.cpu_cores_logical} logicos")
    print(f"  RAM           : {hw.ram_total_gb:.1f} GB ({hw.ram_type})")
    print(f"  GPU           : {hw.gpu_model}")
    print(f"  backend       : {hw.gpu_backend}")
    if hw.gpu_vram_gb > 0:
        print(f"  VRAM          : {hw.gpu_vram_gb:.1f} GB")
    print(f"  pais          : {hw.country_iso}")
    print(f"  factor CO2    : {hw.carbon_intensity_g_kwh} gCO2eq/kWh")
    print(f"  Python        : {hw.python_version}")
    print(f"  llama.cpp     : {hw.llama_cpp_version}")
