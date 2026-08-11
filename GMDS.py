#!/usr/bin/env python3
"""
GMDS.py -- Multi-Layer Glass Volumetric 3D Display System Digital Twin

A physically-grounded, to-scale interactive 3D model of a multi-layer glass
volumetric 3D display (desktop 32" class, 10-layer hybrid stack).
Architecture mirrors LS.py/SSF.py: numpy+pygame software renderer with
painter's algorithm, flat shading, backface culling, orbit/pan/zoom camera.

Usage:
  python GMDS.py                  # interactive 3D viewer
  python GMDS.py --selftest       # headless sanity check
  python GMDS.py --feasibility    # optical/thermal/compute report
  python GMDS.py --export-obj     # write OBJ mesh to ./export/

Requires: numpy, pygame
"""
import argparse, math, sys, os, time
from functools import lru_cache
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

# === CONSTANTS (SI) ===
c_light = 2.998e8
h_planck = 6.626e-34
KB = 1.380649e-23
SIGMA_SB = 5.670374e-8
MM = 1e-3
UM = 1e-6
NM = 1e-9

# =============================================================================
# SECTION 1 -- DIMENSIONS (true physical scale, SI metres)
# =============================================================================
# Desktop-class 32" volumetric TV: 10 hybrid layers, 150 mm total depth.
# 16:9 aspect, panel 699 x 393 mm active area.

DIMS = {
    "panel_diag_in": 32.0,
    "panel_w_m": 0.6989,
    "panel_h_m": 0.3931,
    "n_layers": 10,
    "total_depth_m": 0.150,
    "front_layer_count": 4,
    "front_spacing_mm": 10.0,
    "rear_layer_count": 6,
    "rear_spacing_mm": 18.33,
    "substrate_t_mm": 1.1,
    "active_oled_t_mm": 0.2,
    "active_pdlc_t_mm": 0.15,
    "active_regd_t_mm": 0.8,
    "ar_coat_t_um": 0.3,
    "gel_t_mm": 0.5,
    "pdlc_layers": [0, 1],
    "oled_layers": [2, 3, 4, 5, 6],
    "regd_layers": [7, 8, 9],
    "bezel_w_mm": 12.0,
    "bezel_depth_mm": 165.0,
    "frame_t_mm": 2.5,
    "stand_w_mm": 280.0,
    "stand_d_mm": 200.0,
    "stand_h_mm": 60.0,
    "stand_neck_w_mm": 50.0,
    "stand_neck_d_mm": 40.0,
    "stand_neck_h_mm": 80.0,
    "gpu_board_w_mm": 200.0,
    "gpu_board_h_mm": 100.0,
    "gpu_board_t_mm": 2.0,
    "driver_pcb_w_mm": 120.0,
    "driver_pcb_h_mm": 60.0,
    "driver_pcb_t_mm": 1.6,
    "n_driver_pcbs": 5,
    "psu_w_mm": 140.0,
    "psu_h_mm": 80.0,
    "psu_d_mm": 40.0,
    "heatsink_w_mm": 180.0,
    "heatsink_h_mm": 50.0,
    "heatsink_d_mm": 30.0,
    "heatsink_fins": 12,
    "fan_d_mm": 80.0,
    "fan_t_mm": 25.0,
    "n_fans": 2,
    "nir_laser_bar_w_mm": 10.0,
    "nir_laser_bar_h_mm": 2.0,
    "nir_laser_bar_d_mm": 5.0,
    "n_nir_bars": 6,
    "n_dp_ports": 3,
    "dp_connector_w_mm": 18.0,
    "dp_connector_h_mm": 8.0,
    "tracker_w_mm": 40.0,
    "tracker_h_mm": 8.0,
    "tracker_d_mm": 8.0,
}

# =============================================================================
# SECTION 1b -- PHYSICS PARAMETERS
# =============================================================================

PHYS = {
    "substrate_n": 1.52,
    "gel_n": 1.50,
    "ar_reflectance_pct": 0.3,
    "bare_reflectance_pct": 4.0,
    "oled_transmission_pct": 70.0,
    "pdlc_transmission_clear_pct": 88.0,
    "pdlc_transmission_scat_pct": 35.0,
    "regd_transmission_pct": 85.0,
    "substrate_transmission_pct": 99.5,
    "nir_wavelength_nm": 980.0,
    "nir_power_per_bar_w": 5.0,
    "nir_wallplug_eff": 0.50,
    "upconversion_eff": 0.12,
    "emission_wavelengths_nm": [540, 660, 450],
    "k_aluminium_w_mk": 237.0,
    "k_glass_w_mk": 1.1,
    "oled_power_per_layer_w": 8.0,
    "pdlc_power_per_layer_w": 2.0,
    "total_electronics_w": 350.0,
    "max_surface_temp_c": 45.0,
    "ambient_temp_c": 25.0,
    "target_fps": 240,
    "min_fps": 144,
    "target_latency_ms": 8.0,
    "max_latency_ms": 15.0,
    "pixel_reuse_pct": 85.0,
    "residual_interval": 4,
    "gpu_tflops": 80.0,
    "gpu_vram_gb": 24.0,
    "gpu_bandwidth_gb_s": 1000.0,
    "dp_bandwidth_gbps": 80.0,
    "bits_per_pixel": 30,
    "resolution_w": 3840,
    "resolution_h": 2160,
    "vsync_jitter_us": 50.0,
    "layer_sync_offset_us": 10.0,
    "tracker_fps": 120,
    "tracker_latency_ms": 4.0,
    "tracker_fov_deg": 90.0,
}

# =============================================================================
# SECTION 1c -- MODEL VARIANTS (different screen types & use cases)
# =============================================================================
# Each variant has its own layer count, depth, material, resolution, and FPS
# target per Goal.md Section 4 (Scalability by Use Case).

MODELS = {
    "desktop_monitor": {
        "name": "Desktop Monitor (32\" lightweight)",
        "panel_diag_in": 32.0,
        "panel_w_m": 0.6989,
        "panel_h_m": 0.3931,
        "n_layers": 10,
        "total_depth_m": 0.150,         # 150 mm -- moderate depth, good 3D
        "layer_type": "hybrid",          # PDLC front + OLED mid + RE rear
        "material": "lightweight",       # thin tempered glass + acrylic
        "substrate_t_mm": 1.1,           # thin borosilicate
        "substrate_material": "borosilicate glass",
        "substrate_density_kg_m3": 2230.0,
        "target_fps": 240,
        "resolution_w": 3840,
        "resolution_h": 2160,
        "n_gpus": 1,
        "gpu_tflops": 80.0,
        "total_mass_kg": 8.5,
        "front_spacing_mm": 10.0,
        "rear_spacing_mm": 18.33,
    },
    "commercial_tv": {
        "name": "Large Commercial TV (65\" durable)",
        "panel_diag_in": 65.0,
        "panel_w_m": 1.428,
        "panel_h_m": 0.803,
        "n_layers": 12,
        "total_depth_m": 0.350,         # 350 mm -- thick, strong depth
        "layer_type": "hybrid",
        "material": "durable",           # laminated glass, polycarbonate hybrid
        "substrate_t_mm": 3.0,           # laminated safety glass
        "substrate_material": "laminated safety glass",
        "substrate_density_kg_m3": 2500.0,
        "target_fps": 165,
        "resolution_w": 7680,
        "resolution_h": 4320,
        "n_gpus": 2,
        "gpu_tflops": 160.0,
        "total_mass_kg": 45.0,
        "front_spacing_mm": 18.0,
        "rear_spacing_mm": 32.0,
    },
    "theater": {
        "name": "Theater-Scale (120\" bullet-resistant, 3 ft deep)",
        "panel_diag_in": 120.0,
        "panel_w_m": 2.655,
        "panel_h_m": 1.494,
        "n_layers": 15,
        "total_depth_m": 0.914,         # 914 mm = 3 feet -- maximum depth = maximum realism
        "layer_type": "hybrid_modular",  # modular sub-stacks, doped glass + projected + PDLC
        "material": "bullet_resistant",  # outer enclosure is bullet-resistant
        "substrate_t_mm": 3.0,           # internal layers: laminated safety glass (lighter)
        "substrate_material": "laminated safety glass",  # enclosure is BR, layers are safety
        "substrate_density_kg_m3": 2500.0,
        "target_fps": 120,
        "resolution_w": 7680,
        "resolution_h": 4320,
        "n_gpus": 4,
        "gpu_tflops": 320.0,
        "total_mass_kg": 680.0,
        "front_spacing_mm": 40.0,
        "rear_spacing_mm": 70.0,
    },
    "portable": {
        "name": "Portable Tablet (12\" ultra-lightweight)",
        "panel_diag_in": 12.0,
        "panel_w_m": 0.2655,
        "panel_h_m": 0.1494,
        "n_layers": 6,
        "total_depth_m": 0.030,         # 30 mm -- thin, lightweight
        "layer_type": "oled_only",       # all transparent OLED, no NIR
        "material": "lightweight_film",  # flexible transparent film + acrylic
        "substrate_t_mm": 0.5,           # ultra-thin flexible film
        "substrate_material": "flexible acrylic film",
        "substrate_density_kg_m3": 1180.0,
        "target_fps": 144,
        "resolution_w": 2560,
        "resolution_h": 1440,
        "n_gpus": 1,
        "gpu_tflops": 20.0,
        "total_mass_kg": 0.9,
        "front_spacing_mm": 4.0,
        "rear_spacing_mm": 6.5,
    },
}

# =============================================================================
# SECTION 1d -- MATERIAL PROPERTIES (per substrate type)
# =============================================================================

MATERIALS = {
    "borosilicate glass": {
        "density_kg_m3": 2230.0,
        "youngs_modulus_gpa": 64.0,
        "thermal_expansion_ppm_k": 3.3,
        "transmission_pct_per_mm": 99.5,
        "refractive_index": 1.52,
        "hardness_mohs": 6.5,
        "max_temp_c": 500.0,
        "cost_usd_per_m2": 80.0,
        "note": "standard optical glass, good clarity, moderate weight",
    },
    "laminated safety glass": {
        "density_kg_m3": 2500.0,
        "youngs_modulus_gpa": 70.0,
        "thermal_expansion_ppm_k": 9.0,
        "transmission_pct_per_mm": 98.5,
        "refractive_index": 1.52,
        "hardness_mohs": 6.0,
        "max_temp_c": 300.0,
        "cost_usd_per_m2": 200.0,
        "note": "PVB interlayer, shatter-resistant, heavier, durable install",
    },
    "bullet-resistant laminated glass": {
        "density_kg_m3": 2600.0,
        "youngs_modulus_gpa": 72.0,
        "thermal_expansion_ppm_k": 9.0,
        "transmission_pct_per_mm": 97.0,
        "refractive_index": 1.53,
        "hardness_mohs": 6.5,
        "max_temp_c": 300.0,
        "cost_usd_per_m2": 1200.0,
        "note": "multi-ply glass+polycarbonate, UL 752 rated, very heavy",
    },
    "flexible acrylic film": {
        "density_kg_m3": 1180.0,
        "youngs_modulus_gpa": 3.2,
        "thermal_expansion_ppm_k": 70.0,
        "transmission_pct_per_mm": 99.2,
        "refractive_index": 1.49,
        "hardness_mohs": 3.0,
        "max_temp_c": 80.0,
        "cost_usd_per_m2": 25.0,
        "note": "ultra-light, flexible, low-cost, not permanent/durable",
    },
}

# =============================================================================
# SECTION 1e -- CORRELATION MATRIX & IMAGE PLACEMENT PIPELINE
# =============================================================================
# The rendering pipeline uses CPU+GPU to compute depth-correlation matrices
# that assign scene content to physical layers. This is the core algorithm
# that makes the 3D effect work: each pixel's depth determines which layer(s)
# it renders on, with blending for inter-layer positions.

PIPELINE = {
    "correlation_matrix_method": "MPI_depth_binning",
    "cpu_precompute_tasks": [
        "scene depth buffer extraction",
        "layer assignment matrix construction (NxM sparse)",
        "temporal coherence delta computation",
        "occlusion ordering per layer",
    ],
    "gpu_realtime_tasks": [
        "per-layer rasterization (multi-target render)",
        "temporal reprojection (motion vectors, pixel reuse)",
        "residual correction (lightweight neural, every N frames)",
        "layer compositing + sync output",
    ],
    "matrix_structure": {
        "rows": "scene_depth_bins",       # discretized depth -> physical layer mapping
        "cols": "physical_layers",         # one per glass plane
        "values": "blend_weight_0_to_1",   # how much of pixel goes to each layer
        "sparsity": "~85% zero (most pixels map to 1-2 layers)",
        "update_rate": "delta-based, full recompute only on scene change",
    },
    "depth_binning_algorithm": (
        "1. Render scene to depth buffer (GPU, single pass)\n"
        "2. Quantize depth into N bins matching physical layer Z-positions\n"
        "3. For each pixel: compute blend weights to nearest 2 layers\n"
        "   (linear interpolation in depth for smooth transitions)\n"
        "4. Build sparse correlation matrix M[pixel, layer] = weight\n"
        "5. GPU scatters pixel colors to layer render targets using M\n"
        "6. Temporal reprojection reuses ~85% of prior frame pixels\n"
        "7. Residual path corrects occlusion/transparency every 4 frames"
    ),
}

# =============================================================================
# SECTION 2 -- COLORS & THEME
# =============================================================================
BG_TOP = (90, 95, 100)
BG_BOT = (50, 54, 58)
C_FRAME = (160, 165, 175)
C_BEZEL = (25, 28, 35)
C_GLASS = (200, 225, 240)
C_OLED = (130, 220, 180)
C_PDLC = (150, 195, 235)
C_REGD = (220, 180, 130)
C_AR_COAT = (180, 210, 255)
C_GEL = (220, 235, 248)
C_GPU = (80, 180, 100)
C_DRIVER = (60, 140, 90)
C_PSU = (90, 95, 105)
C_HEATSINK = (200, 205, 215)
C_FAN = (120, 125, 135)
C_NIR = (180, 60, 60)
C_CONNECTOR = (50, 55, 65)
C_STAND = (140, 145, 155)
C_TRACKER = (70, 70, 80)
C_TEXT = (224, 230, 238)
C_TEXT_DIM = (150, 160, 175)
C_ACCENT = (90, 200, 255)
C_GOOD = (90, 220, 130)
C_WARN = (255, 200, 60)
C_BAD = (255, 90, 90)
C_PANEL = (16, 20, 28)
C_PANEL_HI = (28, 36, 50)

# =============================================================================
# SECTION 3 -- MINI 3D ENGINE
# =============================================================================

def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)

def rot_x(a):
    cq, sq = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,cq,-sq],[0,sq,cq]], dtype=float)

def rot_y(a):
    cq, sq = math.cos(a), math.sin(a)
    return np.array([[cq,0,sq],[0,1,0],[-sq,0,cq]], dtype=float)

def rot_z(a):
    cq, sq = math.cos(a), math.sin(a)
    return np.array([[cq,-sq,0],[sq,cq,0],[0,0,1]], dtype=float)

def _mix(c1, c2, t):
    return (int(c1[0]+(c2[0]-c1[0])*t), int(c1[1]+(c2[1]-c1[1])*t), int(c1[2]+(c2[2]-c1[2])*t))


class Mesh:
    """Vertices + polygon faces with base color. Coords in metres."""
    def __init__(self, verts, faces, color, name="", group="static",
                 pivot=(0.,0.,0.), spin=0., alpha=255):
        self.verts = np.asarray(verts, dtype=float)
        self.faces = faces
        self.color = color
        self.name = name
        self.group = group
        self.pivot = np.asarray(pivot, dtype=float)
        self.spin = spin
        self.alpha = alpha
        # pre-compute world verts for static meshes (no spin)
        self._static_world = self.verts + self.pivot if not spin else None

    def world_verts(self, angle=0.0):
        if self._static_world is not None:
            return self._static_world
        v = self.verts
        if self.spin:
            v = v @ rot_z(angle * self.spin).T
        return v + self.pivot


def _solid_cylinder(r, z0, z1, seg=28):
    seg = max(6, int(seg))
    verts, faces = [], []
    ang = np.linspace(0, 2*np.pi, seg, endpoint=False)
    for z in (z0, z1):
        for a in ang:
            verts.append((r*math.cos(a), r*math.sin(a), z))
    c0 = len(verts); verts.append((0,0,z0))
    c1 = len(verts); verts.append((0,0,z1))
    for i in range(seg):
        a, b = i, (i+1)%seg
        faces.append((a, b, seg+b, seg+a))
        faces.append((c0, b, a))
        faces.append((c1, seg+a, seg+b))
    return verts, faces

def _annulus(r_out, r_in, z0, z1, seg=32):
    seg = max(6, int(seg))
    verts, faces = [], []
    ang = np.linspace(0, 2*np.pi, seg, endpoint=False)
    for z in (z0, z1):
        for a in ang:
            verts.append((r_out*math.cos(a), r_out*math.sin(a), z))
        for a in ang:
            verts.append((r_in*math.cos(a), r_in*math.sin(a), z))
    def oo(layer, i): return layer*(2*seg)+(i%seg)
    def ii(layer, i): return layer*(2*seg)+seg+(i%seg)
    for i in range(seg):
        faces.append((oo(0,i), oo(0,i+1), oo(1,i+1), oo(1,i)))
        faces.append((ii(0,i), ii(1,i), ii(1,i+1), ii(0,i+1)))
        faces.append((oo(0,i), ii(0,i), ii(0,i+1), oo(0,i+1)))
        faces.append((oo(1,i), oo(1,i+1), ii(1,i+1), ii(1,i)))
    return verts, faces

def _box(cx, cy, cz, sx, sy, sz):
    hx, hy, hz = sx/2, sy/2, sz/2
    v = [(cx-hx,cy-hy,cz-hz),(cx+hx,cy-hy,cz-hz),
         (cx+hx,cy+hy,cz-hz),(cx-hx,cy+hy,cz-hz),
         (cx-hx,cy-hy,cz+hz),(cx+hx,cy-hy,cz+hz),
         (cx+hx,cy+hy,cz+hz),(cx-hx,cy+hy,cz+hz)]
    f = [(0,1,2,3),(4,7,6,5),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)]
    return v, f

def _combine(chunks):
    verts, faces = [], []
    for v, f in chunks:
        base = len(verts)
        verts.extend(v)
        faces.extend([tuple(i+base for i in face) for face in f])
    return verts, faces

def _translate(vf, off):
    v, f = vf
    ox, oy, oz = off
    return [(x+ox, y+oy, z+oz) for x,y,z in v], f


class Part:
    """Named component with meshes, specs, and explode direction."""
    def __init__(self, key, name, meshes, specs, order, explode):
        self.key = key
        self.name = name
        self.meshes = meshes
        self.specs = specs
        self.order = order
        self.explode = np.asarray(explode, dtype=float)


# =============================================================================
# SECTION 4 -- PHYSICS (optics, thermal, compute pipeline)
# =============================================================================

def fresnel_reflectance(n1, n2):
    """Normal-incidence Fresnel: R = ((n1-n2)/(n1+n2))^2"""
    return ((n1 - n2) / (n1 + n2))**2

def ar_coated_reflectance(bare_r, ar_factor=0.075):
    """Multi-layer AR coating reduces bare R by ~13x."""
    return bare_r * ar_factor

def layer_z_positions(n_layers):
    """Z positions (depth from front) of each layer. Denser near viewer."""
    positions = []
    z = 0.0
    for i in range(n_layers):
        positions.append(z)
        if i < DIMS["front_layer_count"] - 1:
            z += DIMS["front_spacing_mm"] * MM
        elif i < n_layers - 1:
            z += DIMS["rear_spacing_mm"] * MM
    return positions

def stack_transmission(n_layers):
    """Overall optical transmission through the full layer stack."""
    t_total = 1.0
    per_layer = []
    bare_r = fresnel_reflectance(1.0, PHYS["substrate_n"])
    ar_r = ar_coated_reflectance(bare_r)
    for i in range(n_layers):
        t_ar = (1.0 - ar_r)**2
        t_sub = PHYS["substrate_transmission_pct"] / 100.0
        if i in DIMS["pdlc_layers"]:
            t_active = PHYS["pdlc_transmission_clear_pct"] / 100.0
            ltype = "PDLC"
        elif i in DIMS["oled_layers"]:
            t_active = PHYS["oled_transmission_pct"] / 100.0
            ltype = "OLED"
        else:
            t_active = PHYS["regd_transmission_pct"] / 100.0
            ltype = "RE-doped"
        t_gel = 0.998 if i < n_layers - 1 else 1.0
        t_layer = t_ar * t_sub * t_active * t_gel
        per_layer.append(dict(layer=i, type=ltype, t_layer=t_layer))
        t_total *= t_layer
    return t_total, per_layer

def total_power_w():
    """Total system power consumption breakdown."""
    oled = len(DIMS["oled_layers"]) * PHYS["oled_power_per_layer_w"]
    pdlc = len(DIMS["pdlc_layers"]) * PHYS["pdlc_power_per_layer_w"]
    nir = (DIMS["n_nir_bars"] * PHYS["nir_power_per_bar_w"]) / PHYS["nir_wallplug_eff"]
    elec = PHYS["total_electronics_w"]
    fans = DIMS["n_fans"] * 5.0
    total = oled + pdlc + nir + elec + fans
    return dict(oled_w=oled, pdlc_w=pdlc, nir_w=nir, electronics_w=elec,
                fan_w=fans, total_w=total)

def thermal_analysis():
    """Steady-state thermal: can heatsink+fans handle the load?"""
    pw = total_power_w()
    hs_area = (DIMS["heatsink_w_mm"] * DIMS["heatsink_h_mm"] * MM * MM *
               DIMS["heatsink_fins"] * 2)
    h_conv = 40.0
    r_conv = 1.0 / (h_conv * hs_area)
    r_cond = (DIMS["heatsink_d_mm"]*MM) / (PHYS["k_aluminium_w_mk"] *
              DIMS["heatsink_w_mm"]*MM * DIMS["heatsink_h_mm"]*MM)
    r_total = r_conv + r_cond
    dt = pw["total_w"] * r_total
    t_surf = PHYS["ambient_temp_c"] + dt
    return dict(total_w=pw["total_w"], dt_k=dt, t_surface_c=t_surf,
                within_spec=(t_surf <= PHYS["max_surface_temp_c"]))

def bandwidth_requirement():
    """Total display bandwidth vs available DP ports."""
    bpf = PHYS["resolution_w"] * PHYS["resolution_h"] * PHYS["bits_per_pixel"]
    total_gbps = bpf * PHYS["target_fps"] * DIMS["n_layers"] / 1e9
    avail = DIMS["n_dp_ports"] * PHYS["dp_bandwidth_gbps"]
    return dict(total_gbps=total_gbps, available_gbps=avail,
                sufficient=(total_gbps <= avail),
                headroom_pct=(avail/total_gbps - 1.0)*100 if total_gbps > 0 else 0)

def rendering_budget():
    """Per-frame compute budget: can the GPU keep up at target FPS?"""
    fps = PHYS["target_fps"]
    frame_ms = 1000.0 / fps
    total_px = PHYS["resolution_w"] * PHYS["resolution_h"] * DIMS["n_layers"]
    fresh_px = total_px * (1.0 - PHYS["pixel_reuse_pct"]/100.0)
    flops = fresh_px * 100.0
    gpu_flops = PHYS["gpu_tflops"] * 1e12
    render_ms = (flops / gpu_flops) * 1000.0
    residual_flops = total_px * 500.0 / PHYS["residual_interval"]
    residual_ms = (residual_flops / gpu_flops) * 1000.0
    total_ms = render_ms + residual_ms
    return dict(frame_ms=frame_ms, render_ms=render_ms, residual_ms=residual_ms,
                total_ms=total_ms, meets_budget=(total_ms < frame_ms),
                utilization_pct=(total_ms/frame_ms)*100)

def crosstalk_analysis():
    """Inter-layer ghost intensity from double-bounce reflections."""
    bare_r = fresnel_reflectance(1.0, PHYS["substrate_n"])
    ar_r = ar_coated_reflectance(bare_r)
    ghost = ar_r * ar_r
    ghost_db = 10.0 * math.log10(ghost) if ghost > 0 else -999
    return dict(ar_reflectance=ar_r, ghost_intensity=ghost, ghost_db=ghost_db,
                acceptable=(ghost_db < -40.0))

def nir_excitation_report():
    """NIR pump performance for rear RE-doped glass layers."""
    n_bars = DIMS["n_nir_bars"]
    total_opt = n_bars * PHYS["nir_power_per_bar_w"]
    total_elec = total_opt / PHYS["nir_wallplug_eff"]
    vis_emit = total_opt * PHYS["upconversion_eff"]
    lum_flux = vis_emit * 300.0
    return dict(total_nir_optical_w=total_opt, total_nir_electrical_w=total_elec,
                visible_emission_w=vis_emit, luminous_flux_lm=lum_flux)

def depth_perception_quality():
    """Depth perception quality metrics."""
    n = DIMS["n_layers"]
    pos = layer_z_positions(n)
    total_d = pos[-1] if pos else 0.0
    view_dist = 0.60
    avg_sp = total_d / (n-1) if n > 1 else 0.0
    ang_arcmin = math.degrees(math.atan(avg_sp / view_dist)) * 60.0 if avg_sp > 0 else 0.0
    quality = min(100, n*10 + total_d/0.30*50)
    return dict(n_layers=n, total_depth_m=total_d, avg_spacing_mm=avg_sp*1000,
                angular_sep_arcmin=ang_arcmin, quality_score=quality)


# =============================================================================
# SECTION 4a -- ADVANCED PHYSICS (full optical + thermal + compute models)
# =============================================================================

def fresnel_angular(n1, n2, theta_deg):
    """Full Fresnel equations for arbitrary incidence angle.
    Returns (Rs, Rp, R_avg) -- s-polarized, p-polarized, unpolarized average."""
    theta = math.radians(theta_deg)
    sin_t = math.sin(theta)
    sin_t2 = (n1 / n2) * sin_t
    if abs(sin_t2) >= 1.0:
        return 1.0, 1.0, 1.0  # total internal reflection
    cos_i = math.cos(theta)
    cos_t = math.sqrt(1.0 - sin_t2**2)
    Rs = ((n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t))**2
    Rp = ((n1*cos_t - n2*cos_i) / (n1*cos_t + n2*cos_i))**2
    return Rs, Rp, (Rs + Rp) / 2.0


def snell_angle(n1, n2, theta_deg):
    """Snell's law: returns refracted angle in degrees, or None if TIR."""
    sin_t = (n1 / n2) * math.sin(math.radians(theta_deg))
    if abs(sin_t) >= 1.0:
        return None
    return math.degrees(math.asin(sin_t))


def beer_lambert(alpha_per_m, thickness_m):
    """Beer-Lambert transmission: T = exp(-alpha * d)."""
    return math.exp(-alpha_per_m * thickness_m)


def absorption_coefficient(transmission_pct_per_mm):
    """Convert transmission %/mm to absorption coefficient (1/m)."""
    t_per_m = (transmission_pct_per_mm / 100.0) ** 1000
    if t_per_m <= 0:
        return float('inf')
    return -math.log(t_per_m)


def luminance_from_power(optical_power_w, emitting_area_m2, solid_angle_sr=math.pi):
    """Luminance in cd/m^2 (nits) from optical power.
    Assumes 683 lm/W peak luminous efficacy at 555nm, derated to ~300 lm/W
    for broadband white/RGB emission."""
    luminous_flux = optical_power_w * 300.0  # lm
    luminance = luminous_flux / (emitting_area_m2 * solid_angle_sr)
    return luminance  # cd/m^2 (nits)


def required_brightness_per_layer(target_nits, n_layers, stack_transmission):
    """How bright must each emissive layer be to achieve target front luminance?
    Accounts for light loss through layers in front of the emitting one."""
    # worst-case: deepest layer must push through all other layers
    # average case: middle layer pushes through ~half the stack
    avg_transmission = stack_transmission ** 0.5  # geometric mean
    required = target_nits / avg_transmission
    return dict(target_nits=target_nits, avg_layer_transmission=avg_transmission,
                required_per_layer_nits=required,
                achievable=(required < 5000))  # 5000 nits is current OLED peak


def mtf_layer_stack(n_layers, layer_spacing_mm, pixel_pitch_um=80.0):
    """Modulation Transfer Function degradation from layer stack parallax.
    At off-axis viewing, content on rear layers is slightly shifted, reducing
    effective MTF. Models the diffraction + geometric blur."""
    pitch_m = pixel_pitch_um * 1e-6
    # geometric blur for 30-degree off-axis viewing
    theta = math.radians(30)
    total_depth = (n_layers - 1) * layer_spacing_mm * MM
    blur_m = total_depth * math.tan(theta) / n_layers
    # Nyquist frequency
    f_nyquist = 1.0 / (2.0 * pitch_m)  # cycles/m
    # MTF at Nyquist (sinc-like roll-off from geometric blur)
    if blur_m > 0:
        arg = math.pi * f_nyquist * blur_m
        mtf_nyquist = abs(math.sin(arg) / arg) if arg != 0 else 1.0
    else:
        mtf_nyquist = 1.0
    return dict(f_nyquist_lp_mm=f_nyquist/1000, blur_um=blur_m*1e6,
                mtf_at_nyquist=mtf_nyquist, acceptable=(mtf_nyquist > 0.3))


def angular_viewing_analysis(n_layers, total_depth_m, viewing_distance_m=0.6):
    """Full angular analysis: viewing cone, accommodation range, parallax."""
    # parallax at edge of display
    half_width = DIMS["panel_w_m"] / 2.0
    max_parallax_angle = math.degrees(math.atan(half_width / viewing_distance_m))
    # accommodation range (dioptres)
    near = viewing_distance_m - total_depth_m/2
    far = viewing_distance_m + total_depth_m/2
    d_near = 1.0/near if near > 0.1 else 10.0
    d_far = 1.0/far if far > 0.1 else 0.1
    accommodation_range = d_near - d_far  # dioptres
    # depth resolution: smallest distinguishable depth step
    depth_res = total_depth_m / (n_layers - 1) if n_layers > 1 else total_depth_m
    # stereoacuity threshold ~20 arcsec = 0.0001 rad
    stereoacuity_depth = 0.0001 * viewing_distance_m**2 / 0.065  # IPD=65mm
    return dict(max_parallax_deg=max_parallax_angle,
                accommodation_range_D=accommodation_range,
                depth_resolution_mm=depth_res*1000,
                stereoacuity_limit_mm=stereoacuity_depth*1000,
                exceeds_stereoacuity=(depth_res < stereoacuity_depth))


@lru_cache(maxsize=8)
def full_optical_report(model_key):
    """Complete optical characterization for a model variant."""
    m = MODELS[model_key]
    mat = MATERIALS[m["substrate_material"]]
    n = m["n_layers"]
    t_total, per_layer = model_stack_transmission(model_key)
    # angular Fresnel at 45 deg
    _, _, r45 = fresnel_angular(1.0, mat["refractive_index"], 45.0)
    # absorption
    alpha = absorption_coefficient(mat["transmission_pct_per_mm"])
    t_sub_beer = beer_lambert(alpha, m["substrate_t_mm"] * MM)
    # brightness
    panel_area = m["panel_w_m"] * m["panel_h_m"]
    # target 500 nits front luminance for desktop, 1000 for theater
    target_nits = 500 if model_key in ("desktop_monitor", "portable") else 1000
    bright = required_brightness_per_layer(target_nits, n, t_total)
    # MTF
    avg_spacing = m["front_spacing_mm"] if n < 8 else (m["front_spacing_mm"] + m["rear_spacing_mm"])/2
    mtf = mtf_layer_stack(n, avg_spacing)
    # angular
    total_d = model_layer_positions(model_key)[-1] if n > 0 else 0
    ang = angular_viewing_analysis(n, total_d)
    return dict(model=m["name"], t_total=t_total, r_45deg=r45,
                alpha_per_m=alpha, t_substrate_beer=t_sub_beer,
                target_nits=target_nits, brightness=bright,
                mtf=mtf, angular=ang, per_layer=per_layer)


# --- GPU Rendering Pipeline Simulation ---

@lru_cache(maxsize=8)
def simulate_mpi_pipeline(model_key, scene_complexity=1.0):
    """Detailed GPU rendering pipeline timing simulation.
    Models each stage of the MPI depth-binning + temporal reprojection pipeline
    with realistic compute costs based on GPU specs."""
    m = MODELS[model_key]
    n = m["n_layers"]
    res_w, res_h = m["resolution_w"], m["resolution_h"]
    total_px = res_w * res_h
    fps = m["target_fps"]
    gpu_tflops = m["gpu_tflops"]
    gpu_bw = PHYS["gpu_bandwidth_gb_s"]  # GB/s
    frame_ms = 1000.0 / fps

    # Stage 1: Scene rendering + depth buffer (standard rasterization)
    # ~50 flops/pixel for basic geometry + lighting
    scene_flops = total_px * 50.0 * scene_complexity
    stage1_ms = (scene_flops / (gpu_tflops * 1e12)) * 1000.0

    # Stage 2: Depth quantization + layer assignment (compute shader)
    # Each pixel: read depth, compute 2 blend weights, write to sparse matrix
    # ~20 flops/pixel + memory bandwidth limited
    assign_flops = total_px * 20.0
    assign_bw_gb = total_px * 16 / 1e9  # 16 bytes per pixel (depth + 2 weights + indices)
    stage2_compute_ms = (assign_flops / (gpu_tflops * 1e12)) * 1000.0
    stage2_bw_ms = (assign_bw_gb / gpu_bw) * 1000.0
    stage2_ms = max(stage2_compute_ms, stage2_bw_ms)

    # Stage 3: Multi-target scatter (write pixels to N layer render targets)
    # Bandwidth-dominated: write total_px * ~2 layers avg * 4 bytes RGBA
    scatter_bw_gb = total_px * 2.0 * 4 / 1e9
    stage3_ms = (scatter_bw_gb / gpu_bw) * 1000.0

    # Stage 4: Temporal reprojection (reuse 85% of prior frame)
    # Only 15% of pixels need full computation
    fresh_fraction = 1.0 - PHYS["pixel_reuse_pct"] / 100.0
    fresh_px = int(total_px * fresh_fraction)
    reproj_flops = fresh_px * 80.0  # motion vector lookup + blend
    stage4_ms = (reproj_flops / (gpu_tflops * 1e12)) * 1000.0

    # Stage 5: Residual correction (every N frames, amortized)
    # Lightweight neural network: ~500 flops/pixel, but only 1/4 frames
    residual_flops = total_px * 500.0 / PHYS["residual_interval"]
    stage5_ms = (residual_flops / (gpu_tflops * 1e12)) * 1000.0

    # Stage 6: Layer compositing + output formatting
    # N layers * total_px * 8 bytes read+write
    composite_bw = n * total_px * 8 / 1e9
    stage6_ms = (composite_bw / gpu_bw) * 1000.0

    # Stage 7: Display output + vsync synchronization
    # Fixed overhead per frame
    stage7_ms = PHYS["vsync_jitter_us"] / 1000.0 + n * PHYS["layer_sync_offset_us"] / 1000.0

    total_ms = stage1_ms + stage2_ms + stage3_ms + stage4_ms + stage5_ms + stage6_ms + stage7_ms

    # Async overlap: stages 4+5 can overlap with 6
    overlap_ms = min(stage4_ms + stage5_ms, stage6_ms) * 0.7
    effective_ms = total_ms - overlap_ms

    return dict(
        frame_budget_ms=frame_ms,
        stage1_scene_ms=stage1_ms,
        stage2_assign_ms=stage2_ms,
        stage3_scatter_ms=stage3_ms,
        stage4_reproj_ms=stage4_ms,
        stage5_residual_ms=stage5_ms,
        stage6_composite_ms=stage6_ms,
        stage7_sync_ms=stage7_ms,
        total_sequential_ms=total_ms,
        async_overlap_ms=overlap_ms,
        effective_ms=effective_ms,
        meets_budget=(effective_ms < frame_ms),
        utilization_pct=(effective_ms / frame_ms) * 100,
        headroom_ms=frame_ms - effective_ms,
        latency_ms=effective_ms + PHYS["tracker_latency_ms"],
        meets_latency=(effective_ms + PHYS["tracker_latency_ms"] < PHYS["max_latency_ms"]),
    )


@lru_cache(maxsize=8)
def dp_link_budget(model_key):
    """DisplayPort link budget: can the interface handle all layers at target FPS?
    Models compression, DSC, and multi-port distribution."""
    m = MODELS[model_key]
    n = m["n_layers"]
    bpp = PHYS["bits_per_pixel"]
    fps = m["target_fps"]
    res = m["resolution_w"] * m["resolution_h"]
    # raw uncompressed bandwidth per layer
    raw_per_layer_gbps = res * bpp * fps / 1e9
    raw_total_gbps = raw_per_layer_gbps * n
    # DSC 1.2a compression: 3:1 effective for visually lossless
    dsc_ratio = 3.0
    compressed_gbps = raw_total_gbps / dsc_ratio
    # available: DP 2.1 UHBR20 = 80 Gbps per port, with overhead ~97% efficiency
    port_bw = PHYS["dp_bandwidth_gbps"] * 0.97
    n_ports_needed = math.ceil(compressed_gbps / port_bw)
    available = DIMS["n_dp_ports"] * port_bw
    return dict(raw_total_gbps=raw_total_gbps,
                dsc_compressed_gbps=compressed_gbps,
                port_bandwidth_gbps=port_bw,
                ports_needed=n_ports_needed,
                ports_available=DIMS["n_dp_ports"],
                sufficient=(compressed_gbps <= available),
                headroom_pct=((available/compressed_gbps) - 1)*100 if compressed_gbps > 0 else 0)


@lru_cache(maxsize=8)
def per_layer_power_model(model_key):
    """Dynamic power model: each layer's power depends on content and type."""
    m = MODELS[model_key]
    n = m["n_layers"]
    layers = []
    total_w = 0.0
    for i in range(n):
        if m["layer_type"] == "oled_only":
            # OLED: power proportional to pixel brightness, avg 50% lit
            active_w = 8.0 * 0.5  # 4W average per layer
            idle_w = 0.5
            ltype = "OLED"
        elif m["layer_type"] in ("hybrid", "hybrid_modular"):
            if i < 2:
                active_w = 2.0; idle_w = 0.3; ltype = "PDLC"
            elif i < n - 3:
                active_w = 6.0 * 0.5; idle_w = 0.5; ltype = "OLED"
            else:
                active_w = 0.0; idle_w = 0.0; ltype = "RE-doped"  # NIR-pumped, separate
        else:
            active_w = 4.0; idle_w = 0.3; ltype = "mixed"
        # average: assume 30% of layers are actively emitting per frame
        avg_w = active_w * 0.30 + idle_w * 0.70
        layers.append(dict(layer=i, type=ltype, active_w=active_w, idle_w=idle_w, avg_w=avg_w))
        total_w += avg_w
    # add NIR if applicable
    nir_w = 0.0
    if m["layer_type"] in ("hybrid", "hybrid_modular"):
        n_re = sum(1 for l in layers if l["type"] == "RE-doped")
        nir_w = n_re * 10.0  # 10W electrical per RE layer for NIR pump
    # GPU power
    gpu_w = m["n_gpus"] * 300.0 * (m["gpu_tflops"] / (m["n_gpus"] * 80.0))  # scale from 300W baseline
    # drivers, fans, misc
    driver_w = n * 3.0
    fan_w = max(2, n) * 2.5
    misc_w = 15.0
    system_total = total_w + nir_w + gpu_w + driver_w + fan_w + misc_w
    return dict(layer_power=layers, layer_total_w=total_w, nir_w=nir_w,
                gpu_w=gpu_w, driver_w=driver_w, fan_w=fan_w, misc_w=misc_w,
                system_total_w=system_total)


@lru_cache(maxsize=8)
def thermal_per_variant(model_key):
    """Full thermal model for a specific variant including heatsink sizing."""
    m = MODELS[model_key]
    pw = per_layer_power_model(model_key)
    total_w = pw["system_total_w"]
    # heatsink sizing: forced air at 40 W/(m^2*K)
    h_conv = 40.0  # W/(m^2*K) with fans
    delta_t_max = PHYS["max_surface_temp_c"] - PHYS["ambient_temp_c"]  # 20K budget
    required_area = total_w / (h_conv * delta_t_max)
    # available rear panel area
    panel_area = m["panel_w_m"] * m["panel_h_m"]
    rear_area = panel_area * 0.7  # 70% of rear usable for cooling
    # do we fit?
    fits = required_area <= rear_area
    actual_dt = total_w / (h_conv * rear_area) if rear_area > 0 else 999
    t_surface = PHYS["ambient_temp_c"] + actual_dt
    # thermal solution recommendation
    if total_w < 100:
        solution = "passive heatsink + small fans"
    elif total_w < 400:
        solution = "finned aluminium heatsink + 2-4 axial fans"
    elif total_w < 1000:
        solution = "liquid cooling loop + radiator"
    else:
        solution = "industrial liquid cooling + heat exchanger"
    return dict(total_w=total_w, required_area_m2=required_area,
                available_area_m2=rear_area, fits_passive=(fits),
                surface_temp_c=t_surface, within_spec=(t_surface <= PHYS["max_surface_temp_c"]),
                delta_t_k=actual_dt, solution=solution, power_breakdown=pw)


@lru_cache(maxsize=8)
def sync_timing_analysis(model_key):
    """Layer synchronization timing: can all layers stay phase-locked?"""
    m = MODELS[model_key]
    n = m["n_layers"]
    fps = m["target_fps"]
    frame_us = 1e6 / fps
    # per-layer sync offset
    layer_offset = PHYS["layer_sync_offset_us"]
    total_skew = layer_offset * (n - 1)
    # acceptable: < 1/4 pixel persistence time
    persistence_us = frame_us  # for sample-and-hold
    acceptable_skew = persistence_us / 4.0
    # vsync jitter
    jitter = PHYS["vsync_jitter_us"]
    worst_case = total_skew + jitter
    return dict(n_layers=n, fps=fps, frame_us=frame_us,
                total_layer_skew_us=total_skew,
                vsync_jitter_us=jitter,
                worst_case_skew_us=worst_case,
                acceptable_skew_us=acceptable_skew,
                synchronized=(worst_case < acceptable_skew))


# =============================================================================
# SECTION 4a-2 -- PROTOTYPING PLAN & BOM
# =============================================================================

PROTOTYPE_STAGES = {
    "stage_a": {
        "name": "Stage A: 4-6 Layer Desktop Prototype",
        "description": "Transparent OLED/LCD hybrid desktop demonstrator",
        "layers": 5,
        "size_in": 24,
        "material": "borosilicate glass",
        "layer_types": ["PDLC", "OLED", "OLED", "OLED", "PDLC"],
        "target_fps": 120,
        "resolution": "1920x1080",
        "gpu": "RTX 4070 (1x)",
        "estimated_cost_usd": 15000,
        "timeline_months": 4,
        "success_criteria": [
            "visible depth separation between front and rear content",
            "stack transmission > 40%",
            "sustained 120 FPS with basic MPI pipeline",
            "no visible tearing or layer desync",
        ],
        "bom": [
            ("Transparent OLED panels 24\" (3x)", 3600),
            ("PDLC modulator panels 24\" (2x)", 1800),
            ("AR-coated borosilicate substrates (5x)", 2000),
            ("Index-matching optical gel", 500),
            ("Aluminium frame + precision spacers", 800),
            ("GPU: RTX 4070", 600),
            ("Multi-output DP splitter + cables", 400),
            ("Driver electronics + PCBs", 2000),
            ("Power supply 500W", 200),
            ("Cooling (heatsink + fans)", 300),
            ("Eye tracker module", 400),
            ("Assembly + calibration labour", 2400),
        ],
        "risks": [
            ("Transparent OLED availability in 24\" format", "medium", "fallback to stacked LCD"),
            ("Layer alignment precision", "low", "precision machined spacers + calibration"),
            ("Brightness insufficient", "medium", "increase OLED drive current, add AR coatings"),
        ],
    },
    "stage_b": {
        "name": "Stage B: 10-12 Layer System with Tracking",
        "description": "Full hybrid stack with residual correction and eye tracking",
        "layers": 10,
        "size_in": 32,
        "material": "borosilicate glass",
        "layer_types": ["PDLC","PDLC","OLED","OLED","OLED","OLED","OLED","RE","RE","RE"],
        "target_fps": 240,
        "resolution": "3840x2160",
        "gpu": "RTX 5090 or A6000 (1x)",
        "estimated_cost_usd": 45000,
        "timeline_months": 8,
        "success_criteria": [
            "continuous depth perception with smooth transitions",
            "residual correction eliminates ghosting/crosstalk",
            "eye tracking enables view-dependent optimization",
            "sustained 240 FPS with full MPI + temporal reprojection",
            "< 8 ms motion-to-photon latency",
        ],
        "bom": [
            ("Transparent OLED panels 32\" (5x)", 12500),
            ("PDLC modulator panels 32\" (2x)", 3000),
            ("RE-doped glass substrates 32\" (3x)", 6000),
            ("NIR laser diode bars 980nm (6x)", 1800),
            ("AR-coated substrates (10x)", 5000),
            ("Index-matching gel (10 gaps)", 1500),
            ("Precision aluminium frame", 2000),
            ("GPU: RTX 5090", 2000),
            ("Multi-stream DP 2.1 hub", 800),
            ("Driver electronics (10-channel)", 4000),
            ("PSU 800W", 300),
            ("Cooling system (liquid)", 1200),
            ("Eye tracker (Tobii-class)", 600),
            ("Calibration + assembly", 4300),
        ],
        "risks": [
            ("10-layer stack transmission too low", "high", "optimize AR coatings, reduce OLED count"),
            ("NIR pump efficiency", "medium", "use higher-efficiency bars, optimize coupling"),
            ("Thermal management at 500W+", "medium", "liquid cooling, rear exhaust design"),
            ("DP bandwidth for 10x 4K@240", "high", "DSC compression + internal tiling"),
        ],
    },
    "stage_c": {
        "name": "Stage C: Modular Theater-Scale Proof",
        "description": "3-foot deep modular volumetric display for cinema/commercial",
        "layers": 15,
        "size_in": 120,
        "material": "laminated safety glass (enclosure: bullet-resistant)",
        "layer_types": ["WG","WG","WG","SC","SC","SC","SC","SC","SC","SC","SC","RE","RE","RE","RE"],
        "target_fps": 120,
        "resolution": "7680x4320",
        "gpu": "4x A100 or H100",
        "estimated_cost_usd": 350000,
        "timeline_months": 18,
        "success_criteria": [
            "viewer perceives continuous 3D volume at 3 ft depth",
            "sustained 120 FPS with 4-GPU pipeline",
            "modular sub-stacks independently serviceable",
            "multi-viewer support via tracking + residual",
            "passes commercial display safety standards",
        ],
        "bom": [
            ("Modular sub-stack assemblies (5x3-layer)", 120000),
            ("Edge-lit waveguide panels 120\" (3x)", 45000),
            ("Scattering film panels 120\" (8x)", 32000),
            ("RE-doped glass panels 120\" (4x)", 28000),
            ("NIR laser arrays (12 bars)", 8000),
            ("Bullet-resistant enclosure", 25000),
            ("4x A100 GPU compute node", 40000),
            ("Custom DP 2.1 distribution", 5000),
            ("Industrial liquid cooling", 8000),
            ("Multi-channel driver system", 15000),
            ("Tracking array (multi-cam)", 3000),
            ("Precision alignment + calibration rig", 12000),
            ("Installation + commissioning", 9000),
        ],
        "risks": [
            ("120\" panel fabrication", "high", "tiled sub-panels with edge blending"),
            ("680 kg structural support", "medium", "dedicated wall mount + floor brace"),
            ("Multi-GPU synchronization", "medium", "NVLink + custom sync firmware"),
            ("Thermal: 1.5 kW dissipation", "high", "industrial cooling, heat exchanger"),
            ("Optical alignment across 3 ft", "high", "modular sub-stacks + active correction"),
        ],
    },
}

MEASUREMENT_METHODS = {
    "fps": "Frame rate measured at display output using hardware frame counter on final sync signal. Must sustain target for 60+ seconds continuously.",
    "latency": "Motion-to-photon measured with high-speed camera (1000+ FPS) tracking physical input to corresponding photon output. Measure at multiple depth layers.",
    "depth_fidelity": "Quantified via structured depth target: place known 3D object, measure perceived depth vs physical depth at each layer boundary. Report depth error in mm.",
    "brightness": "Luminance measured with calibrated spot meter (Konica Minolta LS-160 or equiv) at 0, 15, 30, 45 degrees. Report cd/m^2 per layer and total.",
    "transmission": "Spectrophotometer (broadband white source behind stack, measure at front). Report per-layer and total T at 400-700nm.",
    "crosstalk": "Display single-pixel dot on one layer, measure phantom intensity on adjacent layers with camera. Report in dB.",
    "multi_viewer": "Two viewers at +-20 degrees simultaneously. Each reports depth quality score. Must maintain >70% of on-axis quality.",
    "uniformity": "9-point luminance measurement grid. Report max/min ratio (must be < 1.3).",
    "color_accuracy": "Measure CIE xy at white point and RGB primaries. Report delta-E 2000 (target < 3).",
}

RISK_REGISTER = {
    "optical_loss": {
        "severity": "high", "probability": "high",
        "impact": "stack transmission below usable threshold",
        "mitigation": "hybrid emissive approach, high-efficiency AR, selective per-pixel addressing",
    },
    "compute_overload": {
        "severity": "high", "probability": "medium",
        "impact": "dropped frames, visible lag",
        "mitigation": "temporal reprojection (85% reuse), residual only every N frames, multi-GPU",
    },
    "thermal_runaway": {
        "severity": "high", "probability": "medium",
        "impact": "component damage, safety hazard",
        "mitigation": "liquid cooling, thermal throttling firmware, redundant temp sensors",
    },
    "layer_desync": {
        "severity": "medium", "probability": "low",
        "impact": "depth planes drift, ghosting",
        "mitigation": "hardware genlock, sub-microsecond sync, watchdog reset",
    },
    "content_pipeline": {
        "severity": "medium", "probability": "high",
        "impact": "no native content available",
        "mitigation": "real-time depth estimation from 2D, MPI from stereo, game engine plugins",
    },
    "manufacturing_yield": {
        "severity": "medium", "probability": "medium",
        "impact": "cost overrun, delays",
        "mitigation": "modular sub-stacks, redundant layers, standardized interfaces",
    },
}


# =============================================================================
# SECTION 4b -- MODEL-VARIANT PHYSICS (per screen type)
# =============================================================================

@lru_cache(maxsize=8)
def model_layer_positions(model_key):
    """Layer Z positions for any model variant."""
    m = MODELS[model_key]
    n = m["n_layers"]
    front_n = min(n // 3, n - 1)  # ~1/3 layers are front-dense
    positions = []
    z = 0.0
    for i in range(n):
        positions.append(z)
        if i < front_n - 1:
            z += m["front_spacing_mm"] * MM
        elif i < n - 1:
            z += m["rear_spacing_mm"] * MM
    return positions


@lru_cache(maxsize=8)
def model_stack_transmission(model_key):
    """Optical transmission for a model variant's entire layer stack.

    KEY PRINCIPLE: All layers are physically transparent at all times.
    The stack is see-through like looking through multiple panes of glass.
    Pixels only become visible when electrically addressed at their depth layer:
      - T-OLED pixels: transparent electrode emits photons only when driven
      - PDLC pixels: liquid crystal scatters only when voltage applied (else clear)
      - RE-Glass voxels: transparent until NIR laser excites rare-earth dopant

    At any given pixel coordinate, typically only 1-2 layers are actively
    rendering content; all other layers are in full transparent pass-through.
    The transmission reported here is the worst-case path through all substrates
    and coatings (the 'optical tax' of having multiple glass sheets)."""
    m = MODELS[model_key]
    mat = MATERIALS[m["substrate_material"]]
    n = m["n_layers"]
    n_ri = mat["refractive_index"]
    bare_r = fresnel_reflectance(1.0, n_ri)
    ar_r = ar_coated_reflectance(bare_r)
    t_sub_per_mm = mat["transmission_pct_per_mm"] / 100.0
    sub_t = m["substrate_t_mm"]
    t_total = 1.0
    per_layer = []
    for i in range(n):
        t_ar = (1.0 - ar_r)**2
        t_sub = t_sub_per_mm ** sub_t
        # layer type determines active-mode transmission
        if m["layer_type"] == "oled_only":
            # transparent OLED: 85% when idle (pass-through), 70% when emitting
            # per-pixel only 1-2 layers emit at once; rest are idle
            t_active = 0.85
            ltype = "OLED"
        elif m["layer_type"] == "hybrid_modular":
            # theater: modular stacks use edge-lit waveguides + scattering
            # panels for front layers, RE-doped glass for rear
            if i < 3:
                t_active = 0.92; ltype = "waveguide"  # edge-lit, very transparent
            elif i < n - 4:
                t_active = 0.88; ltype = "scatter"  # scattering film (idle=clear)
            else:
                t_active = 0.90; ltype = "RE-doped"  # thick but fewer absorbers
        elif m["layer_type"] == "hybrid":
            if i < 2:
                t_active = 0.92; ltype = "PDLC"  # clear-state PDLC
            elif i < n - 3:
                t_active = 0.85; ltype = "OLED"  # idle OLED is mostly clear
            else:
                t_active = 0.88; ltype = "RE-doped"
        else:
            t_active = 0.80; ltype = "mixed"
        t_gel = 0.998 if i < n - 1 else 1.0
        t_layer = t_ar * t_sub * t_active * t_gel
        per_layer.append(dict(layer=i, type=ltype, t_layer=t_layer))
        t_total *= t_layer
    return t_total, per_layer


@lru_cache(maxsize=8)
def model_rendering_budget(model_key):
    """Rendering pipeline budget for a specific model variant."""
    m = MODELS[model_key]
    fps = m["target_fps"]
    frame_ms = 1000.0 / fps
    total_px = m["resolution_w"] * m["resolution_h"] * m["n_layers"]
    fresh_px = total_px * (1.0 - PHYS["pixel_reuse_pct"]/100.0)
    flops = fresh_px * 100.0
    gpu_flops = m["gpu_tflops"] * 1e12
    render_ms = (flops / gpu_flops) * 1000.0
    residual_flops = total_px * 500.0 / PHYS["residual_interval"]
    residual_ms = (residual_flops / gpu_flops) * 1000.0
    total_ms = render_ms + residual_ms
    return dict(frame_ms=frame_ms, render_ms=render_ms, residual_ms=residual_ms,
                total_ms=total_ms, meets_budget=(total_ms < frame_ms),
                utilization_pct=(total_ms/frame_ms)*100,
                n_gpus=m["n_gpus"], gpu_tflops=m["gpu_tflops"])


@lru_cache(maxsize=8)
def model_correlation_matrix_size(model_key):
    """Compute the correlation/assignment matrix dimensions for a variant."""
    m = MODELS[model_key]
    total_pixels = m["resolution_w"] * m["resolution_h"]
    n_layers = m["n_layers"]
    # sparse matrix: each pixel maps to at most 2 layers
    nnz = total_pixels * 2  # non-zero entries
    matrix_bytes = nnz * 4  # float32 weights
    index_bytes = nnz * 4   # layer indices (int32)
    total_mb = (matrix_bytes + index_bytes) / (1024**2)
    # update cost: delta-based, typically 15% of pixels change per frame
    delta_fraction = 1.0 - PHYS["pixel_reuse_pct"]/100.0
    update_pixels = int(total_pixels * delta_fraction)
    return dict(total_pixels=total_pixels, n_layers=n_layers,
                nnz=nnz, matrix_mb=total_mb, delta_pixels=update_pixels,
                full_recompute_ms=total_mb / 50.0,  # ~50 GB/s CPU bandwidth
                delta_update_ms=total_mb * delta_fraction / 50.0)


@lru_cache(maxsize=8)
def model_mass_estimate(model_key):
    """Estimate total display mass from glass layers + frame."""
    m = MODELS[model_key]
    mat = MATERIALS[m["substrate_material"]]
    panel_area = m["panel_w_m"] * m["panel_h_m"]
    glass_vol = panel_area * m["substrate_t_mm"] * MM * m["n_layers"]
    glass_mass = glass_vol * mat["density_kg_m3"]
    # frame: aluminium, ~15% of total for desktop, more for heavy
    frame_fraction = 0.15 if m["material"] == "lightweight" else 0.25
    electronics_kg = 3.0 * m["n_gpus"]
    total = glass_mass * (1.0 + frame_fraction) + electronics_kg
    return dict(glass_mass_kg=glass_mass, frame_mass_kg=glass_mass*frame_fraction,
                electronics_kg=electronics_kg, total_kg=total,
                specified_kg=m["total_mass_kg"])


@lru_cache(maxsize=8)
def model_depth_quality(model_key):
    """Depth perception quality for a variant.
    Quality is a weighted combination of:
    - Depth (thicker = more parallax separation, more real)
    - Layer count (more layers = smoother depth transitions)
    - Layer density (layers per cm of depth = finer depth resolution)
    Each factor is scored 0-100 and weighted by perceptual impact."""
    m = MODELS[model_key]
    pos = model_layer_positions(model_key)
    total_d = pos[-1] if pos else 0.0
    n = m["n_layers"]
    avg_sp = total_d / (n-1) if n > 1 else 0.0
    # depth score: 100mm already gives good parallax at 60cm viewing
    # diminishing returns past ~500mm
    depth_score = min(100, (1.0 - math.exp(-total_d / 0.20)) * 100)
    # layer score: 6 layers = decent, 10+ = excellent
    layer_score = min(100, (1.0 - math.exp(-n / 6.0)) * 100)
    # density: layers per 100mm of depth -- higher is smoother
    density = (n / (total_d * 10)) if total_d > 0 else 0
    density_score = min(100, density / 1.5 * 100)  # 1.5 layers/cm = 100
    quality = depth_score * 0.40 + layer_score * 0.35 + density_score * 0.25
    return dict(model=m["name"], n_layers=n, total_depth_m=total_d,
                total_depth_mm=total_d*1000, avg_spacing_mm=avg_sp*1000,
                depth_score=depth_score, layer_score=layer_score,
                density_score=density_score, quality_score=quality)


# =============================================================================
# SECTION 5 -- GEOMETRY BUILDERS (to-scale 3D model)
# =============================================================================

def build_model_variant(model_key):
    """Build 3D geometry for ANY model variant. Returns (parts, depth).
    Adapts layer count, spacing, material thickness, panel size, and
    electronics to the chosen model."""
    m = MODELS[model_key]
    mat = MATERIALS[m["substrate_material"]]
    parts = []
    pw = m["panel_w_m"]
    ph = m["panel_h_m"]
    bezel = 0.012 if m["material"] != "bullet_resistant" else 0.025
    frame_t = 0.0025 if m["material"] == "lightweight" else 0.005
    total_w = pw + 2*bezel
    total_h = ph + 2*bezel
    n_layers = m["n_layers"]
    lpos = model_layer_positions(model_key)
    total_depth = lpos[-1] + 0.020 if lpos else 0.05
    frame_depth = total_depth + 0.040  # electronics bay behind stack

    # substrate color varies by material (all tinted light to convey transparency)
    mat_colors = {
        "borosilicate glass": C_GLASS,
        "laminated safety glass": (195, 215, 228),
        "bullet-resistant laminated glass": (185, 200, 215),
        "flexible acrylic film": (225, 238, 248),
    }
    glass_color = mat_colors.get(m["substrate_material"], C_GLASS)

    # --- bezel ---
    bz = []
    bz.append(_box(0, ph/2+bezel/2, frame_depth/2, total_w, bezel, frame_depth))
    bz.append(_box(0, -(ph/2+bezel/2), frame_depth/2, total_w, bezel, frame_depth))
    bz.append(_box(-(pw/2+bezel/2), 0, frame_depth/2, bezel, ph, frame_depth))
    bz.append(_box(pw/2+bezel/2, 0, frame_depth/2, bezel, ph, frame_depth))
    v, f = _combine(bz)
    bezel_color = C_BEZEL if m["material"] != "bullet_resistant" else (40, 45, 55)
    parts.append(Part("bezel", f"Bezel ({m['material']})",
                       [Mesh(v, f, bezel_color, "bezel")],
                       [f"{total_w*1000:.0f} x {total_h*1000:.0f} mm",
                        f"material: {m['substrate_material']}"],
                       order=0, explode=(0, 0, -0.15)))

    # --- rear panel ---
    v, f = _box(0, 0, frame_depth, total_w-2*frame_t, total_h-2*frame_t, frame_t)
    parts.append(Part("back_panel", "Rear Panel",
                       [Mesh(v, f, C_FRAME, "back")],
                       [f"depth {frame_depth*1000:.0f} mm"],
                       order=1, explode=(0, 0, 0.15)))

    # --- display layers ---
    # TRANSPARENCY MODEL: All layers are physically transparent at all times.
    # Glass substrate: >99% transmission per layer (AR-coated both sides).
    # Active layers are transparent UNTIL a pixel is addressed:
    #   - OLED: pixel electrodes selectively emit light; unlit pixels are clear
    #   - PDLC: voltage switches pixel from scattering to clear; default = clear
    #   - RE-Glass: transparent until NIR laser excites specific voxel region
    # A viewer sees through the entire stack at all times -- only the rendered
    # pixels at their assigned depth layer become visible as floating light points.
    sub_t = m["substrate_t_mm"] * MM
    for i in range(n_layers):
        z = lpos[i] + 0.010
        if m["layer_type"] == "oled_only":
            at = 0.2 * MM; ac = C_OLED; lt = "T-OLED"
            desc = "Transparent OLED (clear until pixel emits)"
        elif m["layer_type"] in ("hybrid", "hybrid_modular"):
            if i < 2:
                at = 0.15*MM; ac = C_PDLC; lt = "PDLC"
                desc = "PDLC (clear default, scatters only when addressed)"
            elif i < n_layers - 3:
                at = 0.2*MM; ac = C_OLED; lt = "T-OLED"
                desc = "Transparent OLED (clear until pixel emits)"
            else:
                at = 0.8*MM; ac = C_REGD; lt = "RE-Glass"
                desc = "RE-doped glass (clear until NIR-excited voxel)"
        else:
            at = 0.2*MM; ac = C_OLED; lt = "T-OLED"
            desc = "Transparent OLED (clear until pixel emits)"

        v_g, f_g = _box(0, 0, z, pw*0.98, ph*0.98, sub_t)
        v_a, f_a = _box(0, 0, z+sub_t/2+at/2, pw*0.96, ph*0.96, at)
        meshes = [Mesh(v_g, f_g, glass_color, f"sub L{i}"),
                  Mesh(v_a, f_a, ac, f"active L{i}")]
        spacing_info = f"{lpos[i]*1000:.1f} mm" if i > 0 else "0 (front)"
        t_per_layer, _ = model_stack_transmission(model_key)
        t_single = t_per_layer ** (1.0/n_layers) if n_layers > 0 else 0.95
        parts.append(Part(f"layer_{i}", f"Layer {i}: {lt} (transparent)",
                           meshes,
                           [desc,
                            f"substrate: {m['substrate_t_mm']} mm {m['substrate_material']}",
                            f"T per layer: ~{t_single*100:.0f}% (clear until pixel active)",
                            f"depth position: {spacing_info}"],
                           order=10+i, explode=(0, 0, -0.08+i*0.035)))

    # --- GPU(s) ---
    for gi in range(m["n_gpus"]):
        gx = -0.08 + gi * 0.10
        gw = 0.12; gh = 0.08
        v, f = _box(gx, 0.02, frame_depth-0.025, gw, gh, 0.002)
        cv, cf = _box(gx, 0.02, frame_depth-0.028, 0.03, 0.03, 0.005)
        parts.append(Part(f"gpu_{gi}", f"GPU {gi} ({m['gpu_tflops']/m['n_gpus']:.0f} TFLOPS)",
                           [Mesh(v, f, C_GPU, f"GPU{gi}"),
                            Mesh(cv, cf, (50,50,60), f"die{gi}")],
                           [f"GPU {gi+1}/{m['n_gpus']}",
                            f"total compute: {m['gpu_tflops']:.0f} TFLOPS"],
                           order=30+gi, explode=(0, 0, 0.12+gi*0.04)))

    # --- stand (not for theater -- wall-mounted) ---
    if model_key != "theater":
        sw = min(0.28, pw * 0.4)
        sd = 0.20 if model_key != "portable" else 0.08
        sh = 0.06 if model_key != "portable" else 0.02
        nw = 0.05 if model_key != "portable" else 0.02
        nh = 0.08 if model_key != "portable" else 0.03
        y_bot = -(total_h/2)
        v_n, f_n = _box(0, y_bot-nh/2, frame_depth/2, nw, nh, sd*0.2)
        v_s, f_s = _box(0, y_bot-nh-sh/2, frame_depth/2, sw, sh, sd)
        parts.append(Part("stand", "Stand",
                           [Mesh(v_n, f_n, C_STAND, "neck"),
                            Mesh(v_s, f_s, C_STAND, "base")],
                           [f"base {sw*1000:.0f} x {sd*1000:.0f} mm"],
                           order=40, explode=(0, -0.15, 0)))
    else:
        # wall mount brackets for theater
        for side in (-1, 1):
            bx = side * (pw/2 + bezel + 0.02)
            v, f = _box(bx, 0, frame_depth/2, 0.05, 0.30, 0.03)
            parts.append(Part(f"mount_{side}", "Wall Mount Bracket",
                               [Mesh(v, f, C_FRAME, "bracket")],
                               ["heavy-duty wall mount, rated for 700+ kg"],
                               order=40, explode=(side*0.10, 0, 0)))

    return parts, frame_depth


def build_display():
    """Full volumetric 3D TV at true physical scale. Returns (parts, depth)."""
    parts = []
    pw = DIMS["panel_w_m"]
    ph = DIMS["panel_h_m"]
    bezel = DIMS["bezel_w_mm"] * MM
    frame_t = DIMS["frame_t_mm"] * MM
    total_w = pw + 2*bezel
    total_h = ph + 2*bezel
    frame_depth = DIMS["bezel_depth_mm"] * MM
    n_layers = DIMS["n_layers"]
    lpos = layer_z_positions(n_layers)

    # --- bezel (dark frame) ---
    bz = []
    bz.append(_box(0, ph/2+bezel/2, frame_depth/2, total_w, bezel, frame_depth))
    bz.append(_box(0, -(ph/2+bezel/2), frame_depth/2, total_w, bezel, frame_depth))
    bz.append(_box(-(pw/2+bezel/2), 0, frame_depth/2, bezel, ph, frame_depth))
    bz.append(_box(pw/2+bezel/2, 0, frame_depth/2, bezel, ph, frame_depth))
    v, f = _combine(bz)
    parts.append(Part("bezel", "Display Bezel", [Mesh(v, f, C_BEZEL, "bezel")],
                       [f"{total_w*1000:.0f} x {total_h*1000:.0f} mm outer",
                        f"bezel width {bezel*1000:.0f} mm"],
                       order=0, explode=(0, 0, -0.15)))

    # --- rear panel ---
    v, f = _box(0, 0, frame_depth, total_w-2*frame_t, total_h-2*frame_t, frame_t)
    parts.append(Part("back_panel", "Rear Panel (ventilated aluminium)",
                       [Mesh(v, f, C_FRAME, "back panel")],
                       [f"housing depth {frame_depth*1000:.0f} mm"],
                       order=1, explode=(0, 0, 0.15)))

    # --- display layers (the core volumetric stack) ---
    # Each layer is physically transparent -- you see straight through the stack.
    # Pixels only become visible when electrically addressed at their depth plane.
    for i in range(n_layers):
        z = lpos[i] + 0.010
        sub_t = DIMS["substrate_t_mm"] * MM
        if i in DIMS["pdlc_layers"]:
            at = DIMS["active_pdlc_t_mm"] * MM
            ac = C_PDLC; lt = "PDLC"
            desc = "PDLC (transparent default; scatters light only at addressed pixels)"
        elif i in DIMS["oled_layers"]:
            at = DIMS["active_oled_t_mm"] * MM
            ac = C_OLED; lt = "T-OLED"
            desc = "Transparent OLED (clear glass until pixel electrode emits light)"
        else:
            at = DIMS["active_regd_t_mm"] * MM
            ac = C_REGD; lt = "RE-Glass"
            desc = f"RE-doped glass (transparent until NIR @ {PHYS['nir_wavelength_nm']:.0f} nm excites voxel)"
        v_g, f_g = _box(0, 0, z, pw*0.98, ph*0.98, sub_t)
        v_a, f_a = _box(0, 0, z+sub_t/2+at/2, pw*0.96, ph*0.96, at)
        meshes = [Mesh(v_g, f_g, C_GLASS, f"sub L{i}"),
                  Mesh(v_a, f_a, ac, f"active L{i}")]
        parts.append(Part(f"layer_{i}", f"Layer {i}: {lt} (transparent)", meshes,
                           [desc,
                            f"~99% clear per layer (AR-coated both sides)",
                            f"depth {lpos[i]*1000:.1f} mm from front"],
                           order=10+i, explode=(0, 0, -0.08+i*0.035)))

    # --- NIR laser bars ---
    nir_chunks = []
    bw = DIMS["nir_laser_bar_w_mm"]*MM
    bh = DIMS["nir_laser_bar_h_mm"]*MM
    bd = DIMS["nir_laser_bar_d_mm"]*MM
    for li in DIMS["regd_layers"]:
        z = lpos[li] + 0.010
        for side in (-1, 1):
            x = side * (pw/2 + bezel/2 - 0.005)
            nir_chunks.append(_box(x, 0, z, bw, bh, bd))
    v, f = _combine(nir_chunks)
    parts.append(Part("nir_bars", f"NIR Laser Bars ({DIMS['n_nir_bars']}x)",
                       [Mesh(v, f, C_NIR, "NIR lasers")],
                       [f"{PHYS['nir_power_per_bar_w']:.0f} W optical each",
                        f"{PHYS['nir_wavelength_nm']:.0f} nm pump wavelength"],
                       order=25, explode=(0.10, 0, 0)))

    # --- GPU board ---
    gw = DIMS["gpu_board_w_mm"]*MM
    gh = DIMS["gpu_board_h_mm"]*MM
    gt = DIMS["gpu_board_t_mm"]*MM
    gz = frame_depth - 0.030
    v, f = _box(0, 0.02, gz, gw, gh, gt)
    cv, cf = _box(0, 0.02, gz-gt/2-0.003, 0.04, 0.04, 0.006)
    parts.append(Part("gpu", f"GPU ({PHYS['gpu_tflops']:.0f} TFLOPS)",
                       [Mesh(v, f, C_GPU, "GPU PCB"), Mesh(cv, cf, (50,50,60), "die")],
                       [f"{PHYS['gpu_vram_gb']:.0f} GB HBM, {PHYS['gpu_bandwidth_gb_s']:.0f} GB/s",
                        "MPI depth-binning + temporal reprojection"],
                       order=30, explode=(0, 0, 0.12)))

    # --- driver PCBs ---
    dc = []
    dw = DIMS["driver_pcb_w_mm"]*MM
    dh = DIMS["driver_pcb_h_mm"]*MM
    dt = DIMS["driver_pcb_t_mm"]*MM
    for i in range(DIMS["n_driver_pcbs"]):
        dx = -0.15 + i*0.08
        dc.append(_box(dx, -0.10, frame_depth-0.015, dw, dh, dt))
    v, f = _combine(dc)
    parts.append(Part("drivers", f"Layer Drivers ({DIMS['n_driver_pcbs']}x)",
                       [Mesh(v, f, C_DRIVER, "drivers")],
                       ["synchronized multi-target output, common vsync"],
                       order=31, explode=(0, -0.08, 0.10)))

    # --- PSU ---
    v, f = _box(0.18, -0.05, frame_depth-0.025,
                 DIMS["psu_w_mm"]*MM, DIMS["psu_h_mm"]*MM, DIMS["psu_d_mm"]*MM)
    parts.append(Part("psu", "Power Supply",
                       [Mesh(v, f, C_PSU, "PSU")],
                       [f"~{total_power_w()['total_w']:.0f} W system draw"],
                       order=32, explode=(0.08, 0, 0.10)))

    # --- heatsink + fans ---
    hsw = DIMS["heatsink_w_mm"]*MM
    hsh = DIMS["heatsink_h_mm"]*MM
    hsd = DIMS["heatsink_d_mm"]*MM
    v_b, f_b = _box(0, 0.12, frame_depth-0.010, hsw, hsh, hsd)
    fc = []
    nf = DIMS["heatsink_fins"]
    for fi in range(nf):
        fx = -hsw/2 + (fi+0.5)*hsw/nf
        fc.append(_box(fx, 0.12, frame_depth-0.010, hsw/nf*0.3, hsh, hsd*1.5))
    v_f, f_f = _combine(fc)
    parts.append(Part("heatsink", "Heatsink (finned aluminium)",
                       [Mesh(v_b, f_b, C_HEATSINK, "hs base"),
                        Mesh(v_f, f_f, C_HEATSINK, "hs fins")],
                       [f"{nf} fins, forced convection"],
                       order=33, explode=(0, 0.08, 0.08)))

    fan_chunks = []
    fd = DIMS["fan_d_mm"]*MM
    ft = DIMS["fan_t_mm"]*MM
    for fi in range(DIMS["n_fans"]):
        fx = -0.06 + fi*0.12
        fan_chunks.append(_box(fx, 0.12, frame_depth+ft/2, fd, fd, ft))
    v, f = _combine(fan_chunks)
    parts.append(Part("fans", f"Fans ({DIMS['n_fans']}x {DIMS['fan_d_mm']:.0f}mm)",
                       [Mesh(v, f, C_FAN, "fans")],
                       [f"rear-exhaust, surface < {PHYS['max_surface_temp_c']:.0f} C"],
                       order=34, explode=(0, 0.08, 0.12)))

    # --- I/O connectors ---
    cc = []
    dpw = DIMS["dp_connector_w_mm"]*MM
    dph = DIMS["dp_connector_h_mm"]*MM
    for ci in range(DIMS["n_dp_ports"]):
        cx = -0.05 + ci*0.025
        cc.append(_box(cx, -0.15, frame_depth+0.005, dpw, dph, 0.012))
    cc.append(_box(0.05, -0.15, frame_depth+0.005, 0.012, 0.012, 0.012))
    v, f = _combine(cc)
    parts.append(Part("connectors", f"I/O ({DIMS['n_dp_ports']}x DP2.1 + Power)",
                       [Mesh(v, f, C_CONNECTOR, "I/O")],
                       [f"{PHYS['dp_bandwidth_gbps']:.0f} Gbps per port"],
                       order=35, explode=(0, -0.06, 0.10)))

    # --- tracker ---
    v, f = _box(0, ph/2+bezel/2, -0.002,
                 DIMS["tracker_w_mm"]*MM, DIMS["tracker_h_mm"]*MM, DIMS["tracker_d_mm"]*MM)
    parts.append(Part("tracker", "Eye Tracking Camera",
                       [Mesh(v, f, C_TRACKER, "tracker")],
                       [f"{PHYS['tracker_fps']} FPS, {PHYS['tracker_latency_ms']:.0f} ms",
                        f"FOV {PHYS['tracker_fov_deg']:.0f} deg"],
                       order=36, explode=(0, 0.08, -0.08)))

    # --- stand ---
    sw = DIMS["stand_w_mm"]*MM
    sd = DIMS["stand_d_mm"]*MM
    sh = DIMS["stand_h_mm"]*MM
    nw = DIMS["stand_neck_w_mm"]*MM
    nd = DIMS["stand_neck_d_mm"]*MM
    nh = DIMS["stand_neck_h_mm"]*MM
    y_bot = -(total_h/2)
    v_n, f_n = _box(0, y_bot-nh/2, frame_depth/2, nw, nh, nd)
    v_s, f_s = _box(0, y_bot-nh-sh/2, frame_depth/2, sw, sh, sd)
    parts.append(Part("stand", "Display Stand",
                       [Mesh(v_n, f_n, C_STAND, "neck"),
                        Mesh(v_s, f_s, C_STAND, "base")],
                       [f"base {sw*1000:.0f} x {sd*1000:.0f} mm"],
                       order=40, explode=(0, -0.15, 0)))

    return parts, frame_depth


def build_layer_closeup():
    """Zoomed 20x cross-section of 4 adjacent OLED layers showing construction.
    All components shown are physically transparent in real life:
    - AR coatings: optically clear anti-reflective thin films
    - Glass substrate: optically polished borosilicate, >99.5% clear
    - Active OLED: transparent electrode matrix; emits only at addressed pixels
    - Optical gel: index-matching, optically invisible when bonded
    The viewer sees through the entire stack at all times."""
    parts = []
    Z = 20.0
    sub_t = DIMS["substrate_t_mm"]*MM*Z
    ar_t = DIMS["ar_coat_t_um"]*UM*Z*50
    pw = 0.08*Z
    ph = 0.05*Z
    z = 0.0
    for i in range(4):
        li = i + 3
        at = DIMS["active_oled_t_mm"]*MM*Z
        ac = C_OLED
        lt = "OLED"
        # AR front
        v, f = _box(0, 0, z+ar_t/2, pw, ph, ar_t)
        parts.append(Part(f"ar_f{i}", f"AR Coat front L{li}",
                           [Mesh(v, f, C_AR_COAT, f"AR")],
                           [f"R < {PHYS['ar_reflectance_pct']:.1f}%"],
                           order=i*10, explode=(0,0,-0.02*i)))
        z += ar_t
        # substrate
        v, f = _box(0, 0, z+sub_t/2, pw*0.98, ph*0.98, sub_t)
        parts.append(Part(f"sub_{i}", f"Substrate L{li} (transparent)",
                           [Mesh(v, f, C_GLASS, "glass")],
                           [f"{DIMS['substrate_t_mm']:.1f} mm borosilicate (>99.5% clear)",
                            "optically polished, see-through"],
                           order=i*10+1, explode=(0,0,-0.02*i)))
        z += sub_t
        # AR rear
        v, f = _box(0, 0, z+ar_t/2, pw, ph, ar_t)
        parts.append(Part(f"ar_r{i}", f"AR Coat rear L{li}",
                           [Mesh(v, f, C_AR_COAT, "AR")],
                           [f"{DIMS['ar_coat_t_um']:.1f} um actual"],
                           order=i*10+2, explode=(0,0,-0.02*i)))
        z += ar_t
        # active
        v, f = _box(0, 0, z+at/2, pw*0.96, ph*0.96, at)
        parts.append(Part(f"act_{i}", f"Active {lt} L{li} (transparent)",
                           [Mesh(v, f, ac, lt)],
                           [f"{DIMS['active_oled_t_mm']:.2f} mm transparent OLED",
                            "clear until pixel addressed; emits at depth"],
                           order=i*10+3, explode=(0,0,-0.02*i)))
        z += at
        # gel gap
        if i < 3:
            spacing = DIMS["front_spacing_mm"]*MM*Z
            gel_t = min(DIMS["gel_t_mm"]*MM*Z, spacing - sub_t - 2*ar_t - at)
            if gel_t > 0:
                v, f = _box(0, 0, z+gel_t/2, pw*0.94, ph*0.94, gel_t)
                parts.append(Part(f"gel_{i}", f"Optical Gel L{li} (transparent)",
                                   [Mesh(v, f, C_GEL, "gel")],
                                   [f"n={PHYS['gel_n']:.2f} index-matching",
                                    "optically invisible bonding layer"],
                                   order=i*10+4, explode=(0,0,-0.02*i)))
                z += spacing - sub_t - 2*ar_t - at
            else:
                z += 0.005
    return parts, z


# =============================================================================
# SECTION 6 -- RENDERER
# =============================================================================

class Renderer:
    """Projects + paints list[Part] with painter's algorithm."""
    def __init__(self, parts, home_az=0.55, home_el=0.30, home_dist=0.55,
                 scale=1.0, center=(0.,0.,0.)):
        self.parts = parts
        self._home = (home_az, home_el, home_dist)
        self.az, self.el, self.dist = home_az, home_el, home_dist
        self.pan = np.array([0.0, 0.0])
        self.light = np.array([0.4, 0.6, 1.0])
        self.light /= np.linalg.norm(self.light)
        self.exploded = False
        self.explode_amt = 0.0
        self.section = False
        self.hovered = None
        self.selected = None
        self.scale = scale
        self.center = np.asarray(center, dtype=float)
        self._cam_dirty = True

    def reset_view(self):
        self.az, self.el, self.dist = self._home
        self.pan = np.array([0.0, 0.0])
        self._cam_dirty = True

    def zoom_at(self, factor, mouse_pos=None, rect=None):
        old = self.dist
        self.dist = max(0.05*self.scale, min(20.0*self.scale, self.dist*factor))
        if old <= 1e-9 or mouse_pos is None or rect is None:
            return
        if not rect.collidepoint(mouse_pos):
            return
        anchor = np.array([mouse_pos[0]-(rect.x+rect.w/2.0),
                           mouse_pos[1]-(rect.y+rect.h/2.0)], dtype=float)
        k = old / self.dist
        self.pan = anchor - (anchor - self.pan) * k

    def orbit(self, dx, dy, fine=False):
        sens = 0.004 if fine else 0.009
        self.az += dx * sens
        self.el = max(-1.55, min(1.55, self.el + dy*sens))
        self._cam_dirty = True

    def pan_by(self, dx, dy, fine=False):
        sens = 0.45 if fine else 1.0
        self.pan += np.array([dx*sens, dy*sens])

    def _get_cam_matrix(self):
        """Cache rotation matrix, recompute only when orbit changes."""
        if self._cam_dirty or not hasattr(self, '_Rcam'):
            self._Rcam = rot_x(self.el) @ rot_y(self.az)
            self._cam_dirty = False
        return self._Rcam

    def tick(self, dt):
        target = 1.0 if self.exploded else 0.0
        self.explode_amt += (target - self.explode_amt) * min(1.0, dt*4.0)

    def active_part(self):
        i = self.selected if self.selected is not None else self.hovered
        return self.parts[i] if i is not None else None

    def render(self, surf, rect, show_labels, label_font, mouse_pos=None):
        """Optimized render: vectorized projection, scalar face loop, painter's algorithm."""
        clip = surf.get_clip(); surf.set_clip(rect)
        cx = rect.x + rect.w * 0.5 + self.pan[0]
        cy = rect.y + rect.h * 0.5 + self.pan[1]
        focal = min(rect.w, rect.h) * 1.05
        Rcam = self._get_cam_matrix()
        lx, ly, lz = float(self.light[0]), float(self.light[1]), float(self.light[2])

        polys = []
        labels = []
        screeninfo = []
        sel_or_hov = self.selected if self.selected is not None else self.hovered

        center = self.center
        dist = self.dist
        explode_amt = self.explode_amt

        for pi, part in enumerate(self.parts):
            off = part.explode * explode_amt - center
            highlight = (pi == sel_or_hov)
            allcam = []
            outline = (255, 210, 120) if highlight else (30, 32, 36)
            for m in part.meshes:
                wv = m.world_verts() + off
                cam = wv @ Rcam.T
                cam[:, 2] += dist
                allcam.append(cam)

                faces = m.faces
                if not faces:
                    continue

                # vectorized screen projection
                zv = cam[:, 2]
                valid_mask = zv > 1e-6
                inv_z = np.where(valid_mask, 1.0 / np.maximum(zv, 1e-6), 0.0)
                sx_arr = cx + focal * cam[:, 0] * inv_z
                sy_arr = cy - focal * cam[:, 1] * inv_z

                col = _mix(m.color, (255, 255, 255), 0.30) if highlight else m.color
                cr, cg, cb = col

                # pre-extract to Python lists for fast scalar indexing
                caml = cam.tolist()
                sx_list = sx_arr.tolist()
                sy_list = sy_arr.tolist()
                z_list = zv.tolist()
                valid_list = valid_mask.tolist()

                for face in faces:
                    # skip faces with any vertex behind camera
                    if not all(valid_list[idx] for idx in face):
                        continue
                    # face normal (inline cross product)
                    i0, i1, i2 = face[0], face[1], face[2]
                    v0 = caml[i0]; v1 = caml[i1]; v2 = caml[i2]
                    ax, ay, az_ = v0[0], v0[1], v0[2]
                    e1x = v1[0] - ax; e1y = v1[1] - ay; e1z = v1[2] - az_
                    e2x = v2[0] - ax; e2y = v2[1] - ay; e2z = v2[2] - az_
                    nx = e1y * e2z - e1z * e2y
                    ny = e1z * e2x - e1x * e2z
                    nz = e1x * e2y - e1y * e2x
                    ln = (nx * nx + ny * ny + nz * nz) ** 0.5
                    if ln < 1e-12:
                        continue
                    if nz > 0:
                        nx, ny, nz = -nx, -ny, -nz
                    d = (nx * lx + ny * ly + nz * lz) / ln
                    shade = 0.45 + 0.55 * (d if d > 0.0 else 0.0)
                    fc = (int(cr * shade), int(cg * shade), int(cb * shade))
                    ds = sum(z_list[idx] for idx in face) / len(face)
                    pts = [(sx_list[idx], sy_list[idx]) for idx in face]
                    polys.append((ds, pts, fc, outline))

            if not allcam:
                continue
            cam_all = np.vstack(allcam)
            cen = cam_all.mean(axis=0)
            if cen[2] > 1e-6:
                safez = np.maximum(cam_all[:, 2], 1e-6)
                scx = cx + focal * cam_all[:, 0] / safez
                scy = cy - focal * cam_all[:, 1] / safez
                pcx = float(cx + focal * cen[0] / cen[2])
                pcy = float(cy - focal * cen[1] / cen[2])
                rad = float(np.max(np.hypot(scx - pcx, scy - pcy))) * 0.55 + 6
                screeninfo.append((pi, pcx, pcy, rad, float(cen[2])))
                if show_labels and label_font:
                    labels.append((float(cen[2]), (pcx, pcy), part.name))

        # sort back-to-front (painter's algorithm)
        polys.sort(key=lambda t: t[0], reverse=True)
        # batch draw
        draw_poly = pygame.draw.polygon
        for _, pts, fc, outline in polys:
            if len(pts) < 3:
                continue
            try:
                draw_poly(surf, fc, pts)
                draw_poly(surf, outline, pts, 1)
            except Exception:
                pass

        if show_labels and label_font:
            labels.sort(key=lambda t: t[0])
            used = []
            for _, (lxp, lyp), text in labels:
                ly2 = lyp
                for uy in used:
                    if abs(ly2 - uy) < 15:
                        ly2 = uy + 15
                used.append(ly2)
                img = label_font.render(text, True, C_ACCENT)
                bg = pygame.Surface((img.get_width() + 6, img.get_height() + 2), pygame.SRCALPHA)
                bg.fill((10, 12, 16, 165))
                surf.blit(bg, (lxp + 4, ly2 - img.get_height() / 2 - 1))
                surf.blit(img, (lxp + 7, ly2 - img.get_height() / 2))

        if mouse_pos is not None:
            mxp, myp = mouse_pos
            best, bestd = None, 1e18
            for pi, pcx, pcy, rad, depth in screeninfo:
                if math.hypot(mxp - pcx, myp - pcy) <= rad and depth < bestd:
                    bestd, best = depth, pi
            self.hovered = best if best is not None and best < len(self.parts) else None

        surf.set_clip(clip)


# =============================================================================
# SECTION 7 -- REPORTS (CLI output)
# =============================================================================

def print_feasibility():
    """Full optical / thermal / compute feasibility report for all model variants."""
    print("=" * 72)
    print("  MULTI-LAYER GLASS VOLUMETRIC 3D DISPLAY -- FEASIBILITY REPORT")
    print("  All Model Variants Comparison")
    print("=" * 72)

    # --- Transparency principle ---
    print("\n" + "=" * 72)
    print("  TRANSPARENCY PRINCIPLE")
    print("=" * 72)
    print("""
  All display layers are physically transparent at all times.
  The entire stack is see-through like looking through a window.

  HOW IT WORKS:
    1. Each layer is a sheet of optically-clear glass (AR-coated, >99% T/layer)
    2. Embedded in each glass sheet is a transparent active matrix:
       - T-OLED: transparent organic electrodes; emit light ONLY when pixel driven
       - PDLC: liquid crystal stays clear by default; scatters only when addressed
       - RE-Glass: rare-earth dopant is invisible until NIR laser excites it
    3. The GPU assigns each 3D pixel to a specific depth layer via the
       correlation matrix (Section: Pipeline below)
    4. Only addressed pixels at their assigned layer become visible as
       floating points of light at real physical depth
    5. All unaddressed pixels on all layers remain fully transparent

  RESULT: Viewer perceives floating 3D imagery suspended in clear glass volume.
  No occlusion artifacts between layers because inactive pixels are invisible.
  Real accommodation cues: eyes physically focus at the layer's true depth.
""")

    # --- Per-variant comparison table ---
    print("=" * 72)
    print("  MODEL VARIANT COMPARISON")
    print("=" * 72)
    print(f"  {'Model':<36s} {'Layers':>6s} {'Depth':>8s} {'Res':>10s} {'FPS':>4s} {'GPUs':>4s} {'Mass':>8s}")
    print(f"  {'-'*36} {'-'*6} {'-'*8} {'-'*10} {'-'*4} {'-'*4} {'-'*8}")
    for key, m in MODELS.items():
        depth_str = f"{m['total_depth_m']*1000:.0f} mm"
        res_str = f"{m['resolution_w']}x{m['resolution_h']}"
        mass_str = f"{m['total_mass_kg']:.1f} kg"
        print(f"  {m['name']:<36s} {m['n_layers']:>6d} {depth_str:>8s} {res_str:>10s} "
              f"{m['target_fps']:>4d} {m['n_gpus']:>4d} {mass_str:>8s}")

    # --- Per-variant optical + compute ---
    for key in MODELS:
        m = MODELS[key]
        print(f"\n{'~'*72}")
        print(f"  [{key.upper()}] {m['name']}")
        print(f"{'~'*72}")

        t_total, per_layer = model_stack_transmission(key)
        print(f"  Optical stack T: {t_total*100:.1f}% through {m['n_layers']} layers")
        print(f"  Material: {m['substrate_material']} @ {m['substrate_t_mm']} mm/layer")

        rb = model_rendering_budget(key)
        print(f"  Rendering: {rb['total_ms']:.3f} ms / {rb['frame_ms']:.2f} ms budget "
              f"({rb['utilization_pct']:.1f}% util, {m['n_gpus']}x GPU)")

        cm = model_correlation_matrix_size(key)
        print(f"  Correlation matrix: {cm['total_pixels']/1e6:.1f} Mpx x {cm['n_layers']} layers, "
              f"{cm['matrix_mb']:.0f} MB sparse, delta update {cm['delta_update_ms']:.2f} ms")

        dq = model_depth_quality(key)
        print(f"  Depth quality: {dq['quality_score']:.0f}/100 "
              f"(depth={dq['total_depth_mm']:.0f} mm, {dq['avg_spacing_mm']:.1f} mm avg spacing)")

        mass = model_mass_estimate(key)
        print(f"  Estimated mass: {mass['total_kg']:.1f} kg "
              f"(glass {mass['glass_mass_kg']:.1f} + frame {mass['frame_mass_kg']:.1f} + elec {mass['electronics_kg']:.1f})")

    # --- Pipeline explanation ---
    print(f"\n{'='*72}")
    print("  IMAGE CORRELATION MATRIX PIPELINE (CPU + GPU)")
    print(f"{'='*72}")
    print(f"  Method: {PIPELINE['correlation_matrix_method']}")
    print(f"\n  CPU pre-compute tasks:")
    for t in PIPELINE["cpu_precompute_tasks"]:
        print(f"    - {t}")
    print(f"\n  GPU real-time tasks:")
    for t in PIPELINE["gpu_realtime_tasks"]:
        print(f"    - {t}")
    print(f"\n  Matrix structure:")
    for k, v in PIPELINE["matrix_structure"].items():
        print(f"    {k}: {v}")
    print(f"\n  Depth-Binning Algorithm:")
    for line in PIPELINE["depth_binning_algorithm"].split("\n"):
        print(f"    {line}")

    # --- Per-variant ADVANCED analysis ---
    print(f"\n{'='*72}")
    print("  ADVANCED PHYSICS PER VARIANT")
    print(f"{'='*72}")
    for key in MODELS:
        m = MODELS[key]
        print(f"\n  [{key.upper()}]")
        # full optical
        opt = full_optical_report(key)
        print(f"    Fresnel R @ 45 deg: {opt['r_45deg']*100:.2f}%")
        print(f"    Beer-Lambert T (substrate): {opt['t_substrate_beer']*100:.2f}%")
        print(f"    Required brightness/layer: {opt['brightness']['required_per_layer_nits']:.0f} nits "
              f"({'achievable' if opt['brightness']['achievable'] else 'EXCEEDS OLED LIMITS'})")
        print(f"    MTF @ Nyquist (30 deg off-axis): {opt['mtf']['mtf_at_nyquist']:.3f} "
              f"({'OK' if opt['mtf']['acceptable'] else 'DEGRADED'})")
        print(f"    Accommodation range: {opt['angular']['accommodation_range_D']:.2f} D")
        print(f"    Depth resolution: {opt['angular']['depth_resolution_mm']:.1f} mm/step")

        # detailed pipeline
        pip = simulate_mpi_pipeline(key)
        print(f"    GPU Pipeline (7-stage):")
        print(f"      S1 scene:     {pip['stage1_scene_ms']:.4f} ms")
        print(f"      S2 assign:    {pip['stage2_assign_ms']:.4f} ms")
        print(f"      S3 scatter:   {pip['stage3_scatter_ms']:.4f} ms")
        print(f"      S4 reproj:    {pip['stage4_reproj_ms']:.4f} ms")
        print(f"      S5 residual:  {pip['stage5_residual_ms']:.4f} ms")
        print(f"      S6 composite: {pip['stage6_composite_ms']:.4f} ms")
        print(f"      S7 sync:      {pip['stage7_sync_ms']:.4f} ms")
        print(f"      Effective:    {pip['effective_ms']:.3f} ms / {pip['frame_budget_ms']:.2f} ms "
              f"({pip['utilization_pct']:.1f}%)")
        print(f"      Latency:      {pip['latency_ms']:.2f} ms "
              f"({'OK' if pip['meets_latency'] else 'EXCEEDS TARGET'})")

        # DP link
        dp = dp_link_budget(key)
        print(f"    DP link: {dp['dsc_compressed_gbps']:.1f} Gbps (DSC 3:1) / "
              f"{dp['ports_available']}x{dp['port_bandwidth_gbps']:.0f} Gbps "
              f"({dp['ports_needed']} ports needed) "
              f"{'OK' if dp['sufficient'] else 'INSUFFICIENT'}")

        # thermal
        th = thermal_per_variant(key)
        print(f"    Thermal: {th['total_w']:.0f} W, {th['surface_temp_c']:.1f} C "
              f"({'OK' if th['within_spec'] else 'OVER SPEC'}) -> {th['solution']}")

        # sync
        sync = sync_timing_analysis(key)
        print(f"    Sync: {sync['worst_case_skew_us']:.1f} us skew / "
              f"{sync['acceptable_skew_us']:.0f} us budget "
              f"({'LOCKED' if sync['synchronized'] else 'NEEDS GENLOCK'})")

    # --- Prototyping plan ---
    print(f"\n{'='*72}")
    print("  STAGED PROTOTYPING PLAN")
    print(f"{'='*72}")
    for stage_key, stage in PROTOTYPE_STAGES.items():
        print(f"\n  {stage['name']}")
        print(f"  {'='*60}")
        print(f"    {stage['description']}")
        print(f"    Layers: {stage['layers']}, Size: {stage['size_in']}\", "
              f"Material: {stage['material']}")
        print(f"    Resolution: {stage['resolution']} @ {stage['target_fps']} FPS")
        print(f"    GPU: {stage['gpu']}")
        print(f"    Timeline: {stage['timeline_months']} months")
        print(f"    Estimated cost: ${stage['estimated_cost_usd']:,}")
        print(f"    Success criteria:")
        for c in stage["success_criteria"]:
            print(f"      - {c}")
        bom_total = sum(cost for _, cost in stage["bom"])
        print(f"    BOM ({len(stage['bom'])} items, ${bom_total:,} total):")
        for item, cost in stage["bom"]:
            print(f"      ${cost:>6,}  {item}")
        print(f"    Risks:")
        for risk, sev, mitigation in stage["risks"]:
            print(f"      [{sev.upper():>6s}] {risk}")
            print(f"               -> {mitigation}")

    # --- Risk register ---
    print(f"\n{'='*72}")
    print("  RISK REGISTER")
    print(f"{'='*72}")
    for rk, rv in RISK_REGISTER.items():
        print(f"  {rk.upper()}: severity={rv['severity']}, probability={rv['probability']}")
        print(f"    Impact: {rv['impact']}")
        print(f"    Mitigation: {rv['mitigation']}")

    # --- Measurement methods ---
    print(f"\n{'='*72}")
    print("  MEASUREMENT & VALIDATION METHODS")
    print(f"{'='*72}")
    for mk, mv in MEASUREMENT_METHODS.items():
        print(f"  {mk}: {mv}")

    # --- Crosstalk ---
    print(f"\n{'='*72}")
    print("  CROSSTALK ANALYSIS (Desktop reference)")
    print(f"{'='*72}")
    ct = crosstalk_analysis()
    print(f"  AR-coated reflectance per surface: {ct['ar_reflectance']*100:.3f}%")
    print(f"  Ghost image intensity (double-bounce): {ct['ghost_db']:.1f} dB")
    print(f"  Acceptable (< -40 dB): {'YES' if ct['acceptable'] else 'NO'}")

    print("\n" + "=" * 72)
    print("  END OF COMPREHENSIVE FEASIBILITY REPORT")
    print("=" * 72)


def export_obj(path="export"):
    """Write OBJ mesh file for the full display model."""
    parts, _ = build_display()
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, "gmds_display.obj")
    vert_offset = 0
    with open(filepath, "w") as fp:
        fp.write("# GMDS.py -- Multi-Layer Glass Volumetric 3D Display\n")
        fp.write(f"# {len(parts)} parts, generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for part in parts:
            fp.write(f"o {part.key}\n")
            for m in part.meshes:
                for vx, vy, vz in m.verts.tolist():
                    fp.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
                for face in m.faces:
                    indices = " ".join(str(idx + 1 + vert_offset) for idx in face)
                    fp.write(f"f {indices}\n")
                vert_offset += len(m.verts)
    print(f"Exported {vert_offset} vertices to {filepath}")


# =============================================================================
# SECTION 8 -- APPLICATION (interactive pygame viewer)
# =============================================================================

SCENES = ["desktop_monitor", "commercial_tv", "theater", "portable", "layers"]
SCENE_TITLES = {
    "desktop_monitor": "DESKTOP (32\" 10L transparent stack, pixels float at depth)",
    "commercial_tv": "COMMERCIAL (65\" 12L see-through volume, per-pixel depth addressing)",
    "theater": "THEATER (120\" 15L clear glass volume, 3ft deep, real focal cues)",
    "portable": "PORTABLE (12\" 6L transparent stack, 30mm depth)",
    "layers": "LAYER CLOSE-UP (all layers transparent until pixel addressed)",
}


class App:
    """Interactive 3D viewer with scene switching, HUD, exploded view."""
    def __init__(self):
        if pygame is None:
            print("ERROR: pygame not installed. pip install pygame"); sys.exit(1)
        pygame.init()
        info = pygame.display.Info()
        self.W = min(1600, info.current_w - 100)
        self.H = min(1000, info.current_h - 100)
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        pygame.display.set_caption("GMDS -- Multi-Layer Glass Volumetric 3D Display")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 13)
        self.font_big = pygame.font.SysFont("consolas", 15, bold=True)

        # build all model variant scenes
        self.renderers = {}
        for key in MODELS:
            parts, depth = build_model_variant(key)
            # scale camera distance based on panel size
            m = MODELS[key]
            cam_dist = max(m["panel_w_m"], m["panel_h_m"]) * 1.6
            self.renderers[key] = Renderer(parts, home_az=0.45, home_el=0.25,
                                           home_dist=cam_dist,
                                           scale=max(0.3, cam_dist/2),
                                           center=(0, 0, depth/2))

        # layer closeup scene
        layer_parts, layer_depth = build_layer_closeup()
        self.renderers["layers"] = Renderer(layer_parts, home_az=0.3, home_el=0.2,
                                            home_dist=layer_depth*1.8,
                                            scale=layer_depth,
                                            center=(0, 0, layer_depth/2))

        self.scene_idx = 0
        self.show_labels = True
        self.show_hud = True
        self.show_info = False
        self.info_scroll = 0
        self.dragging = False
        self.panning = False
        self.running = True
        self._fps_history = []
        self._frame_count = 0

    @property
    def scene(self):
        return SCENES[self.scene_idx]

    @property
    def renderer(self):
        return self.renderers[self.scene]

    def current_model_key(self):
        """Return the model key if current scene is a model variant, else None."""
        s = self.scene
        return s if s in MODELS else None

    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False
            elif ev.type == pygame.VIDEORESIZE:
                self.W, self.H = ev.w, ev.h
                self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self.running = False
                elif ev.key == pygame.K_TAB:
                    self.scene_idx = (self.scene_idx + 1) % len(SCENES)
                elif ev.key == pygame.K_e:
                    self.renderer.exploded = not self.renderer.exploded
                elif ev.key == pygame.K_l:
                    self.show_labels = not self.show_labels
                elif ev.key == pygame.K_h:
                    self.show_hud = not self.show_hud
                elif ev.key == pygame.K_r:
                    self.renderer.reset_view()
                elif ev.key == pygame.K_s:
                    self.renderer.section = not self.renderer.section
                elif ev.key == pygame.K_i:
                    self.show_info = not self.show_info
                    self.info_scroll = 0
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    self.dragging = True
                elif ev.button == 3:
                    self.panning = True
                elif ev.button == 4:
                    if self.show_info:
                        self.info_scroll = max(0, self.info_scroll - 3)
                    else:
                        self.renderer.zoom_at(0.9, ev.pos, pygame.Rect(0,0,self.W,self.H))
                elif ev.button == 5:
                    if self.show_info:
                        self.info_scroll += 3
                    else:
                        self.renderer.zoom_at(1.1, ev.pos, pygame.Rect(0,0,self.W,self.H))
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1:
                    self.dragging = False
                    if self.renderer.hovered is not None:
                        self.renderer.selected = self.renderer.hovered
                    else:
                        self.renderer.selected = None
                elif ev.button == 3:
                    self.panning = False
            elif ev.type == pygame.MOUSEMOTION:
                if self.dragging:
                    self.renderer.orbit(ev.rel[0], ev.rel[1])
                elif self.panning:
                    self.renderer.pan_by(ev.rel[0], ev.rel[1])

    def draw_bg(self):
        # pre-render gradient as surface, only regenerate on resize
        if not hasattr(self, '_bg_surf') or self._bg_surf.get_size() != (self.W, self.H):
            self._bg_surf = pygame.Surface((self.W, self.H))
            arr = np.zeros((self.H, self.W, 3), dtype=np.uint8)
            t = np.linspace(0, 1, self.H)[:, None]
            for c in range(3):
                col = (BG_TOP[c] + (BG_BOT[c] - BG_TOP[c]) * t).astype(np.uint8)
                arr[:, :, c] = col
            pygame.surfarray.blit_array(self._bg_surf, arr.transpose(1, 0, 2))
        self.screen.blit(self._bg_surf, (0, 0))

    def draw_hud(self):
        if not self.show_hud:
            return
        # FPS counter
        self._frame_count += 1
        fps = self.clock.get_fps()
        self._fps_history.append(fps)
        if len(self._fps_history) > 60:
            self._fps_history.pop(0)
        avg_fps = sum(self._fps_history) / len(self._fps_history) if self._fps_history else 0

        # title bar
        title = SCENE_TITLES.get(self.scene, "")
        img = self.font_big.render(title, True, C_ACCENT)
        self.screen.blit(img, (10, 8))
        # FPS display top-right
        fps_str = f"FPS: {avg_fps:.0f}"
        fps_color = C_GOOD if avg_fps > 50 else (C_WARN if avg_fps > 30 else C_BAD)
        img = self.font.render(fps_str, True, fps_color)
        self.screen.blit(img, (self.W - 80, 8))

        # controls help + transparency info
        lines = [
            "All layers transparent -- pixels only visible when addressed at depth",
            "TAB: scene | E: explode | L: labels | H: HUD | I: info/blueprint | R: reset",
            "LMB: orbit | RMB: pan | Scroll: zoom | Click: inspect part",
        ]
        for i, line in enumerate(lines):
            color = C_ACCENT if i == 0 else C_TEXT_DIM
            img = self.font.render(line, True, color)
            self.screen.blit(img, (10, self.H - 50 + i*16))

        # part info panel
        part = self.renderer.active_part()
        if part:
            px, py = self.W - 340, 10
            pw2 = 330
            ph2 = 20 + len(part.specs)*16 + 10
            panel = pygame.Surface((pw2, ph2), pygame.SRCALPHA)
            panel.fill((*C_PANEL, 200))
            pygame.draw.rect(panel, (*C_PANEL_HI, 230), panel.get_rect(), 1)
            self.screen.blit(panel, (px, py))
            img = self.font_big.render(part.name, True, C_ACCENT)
            self.screen.blit(img, (px+8, py+6))
            for si, spec in enumerate(part.specs):
                img = self.font.render(spec, True, C_TEXT)
                self.screen.blit(img, (px+8, py+24+si*16))

        # physics sidebar for model variant scenes (cached to avoid per-frame recompute)
        mk = self.current_model_key()
        if mk:
            if not hasattr(self, '_hud_cache_key') or self._hud_cache_key != mk:
                self._hud_cache_key = mk
                self._hud_t_total, _ = model_stack_transmission(mk)
                self._hud_rb = model_rendering_budget(mk)
                self._hud_dq = model_depth_quality(mk)
                self._hud_pip = simulate_mpi_pipeline(mk)
                self._hud_th = thermal_per_variant(mk)
                self._hud_sync = sync_timing_analysis(mk)
                self._hud_dp = dp_link_budget(mk)
            t_total = self._hud_t_total
            rb = self._hud_rb
            dq = self._hud_dq
            pip = self._hud_pip
            th = self._hud_th
            sync = self._hud_sync
            dpb = self._hud_dp
            sx, sy = 10, 40
            m = MODELS[mk]
            metrics = [
                (f"Optical T: {t_total*100:.1f}%", t_total,
                 C_GOOD if t_total > 0.5 else (C_WARN if t_total > 0.2 else C_BAD)),
                (f"GPU: {pip['effective_ms']:.2f}/{pip['frame_budget_ms']:.1f} ms ({pip['utilization_pct']:.0f}%)",
                 pip['utilization_pct']/100,
                 C_GOOD if pip['meets_budget'] else C_BAD),
                (f"Latency: {pip['latency_ms']:.1f} ms",
                 pip['latency_ms']/15.0,
                 C_GOOD if pip['meets_latency'] else C_BAD),
                (f"Thermal: {th['total_w']:.0f} W / {th['surface_temp_c']:.0f} C",
                 th['surface_temp_c']/60.0,
                 C_GOOD if th['within_spec'] else C_BAD),
                (f"Depth: {dq['quality_score']:.0f}/100 ({m['n_layers']}L, {m['total_depth_m']*1000:.0f}mm)",
                 dq['quality_score']/100, C_GOOD if dq['quality_score'] > 60 else C_WARN),
                (f"DP: {dpb['dsc_compressed_gbps']:.0f} Gbps ({dpb['ports_needed']}p needed)",
                 min(1.0, dpb['dsc_compressed_gbps']/(dpb['ports_available']*dpb['port_bandwidth_gbps'])),
                 C_GOOD if dpb['sufficient'] else C_BAD),
                (f"Sync: {sync['worst_case_skew_us']:.0f} us skew",
                 sync['worst_case_skew_us']/sync['acceptable_skew_us'],
                 C_GOOD if sync['synchronized'] else C_WARN),
                (f"Stack: {m['n_layers']}L transparent (clear until pixel active)",
                 0.95, C_ACCENT),
                (f"{m['substrate_material'][:32]}",
                 0.5, C_TEXT_DIM),
            ]
            for i, (label, frac, color) in enumerate(metrics):
                y = sy + i*18
                bw2 = 260
                pygame.draw.rect(self.screen, C_PANEL_HI, (sx, y, bw2, 14))
                fw = max(0, min(bw2, int(bw2 * clamp(frac))))
                pygame.draw.rect(self.screen, color, (sx, y, fw, 14))
                pygame.draw.rect(self.screen, (10,12,16), (sx, y, bw2, 14), 1)
                img = self.font.render(label, True, C_TEXT)
                self.screen.blit(img, (sx+4, y-1))

    def draw_info(self):
        """Full-screen info/blueprint overlay with scrolling."""
        if not self.show_info:
            return
        # semi-transparent dark overlay
        overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        overlay.fill((10, 14, 20, 230))
        self.screen.blit(overlay, (0, 0))

        # info content
        info_lines = [
            ("MULTI-LAYER GLASS VOLUMETRIC 3D DISPLAY", True),
            ("Engineering Blueprint & Fabrication Guide", True),
            ("", False),
            ("=" * 64, False),
            ("HOW IT WORKS", True),
            ("=" * 64, False),
            ("", False),
            ("This display creates true 3D images by stacking multiple", False),
            ("transparent glass layers, each containing an addressable pixel", False),
            ("matrix. ALL layers are physically see-through at all times.", False),
            ("Pixels only become visible when electrically activated at their", False),
            ("specific depth layer, creating floating points of light in space.", False),
            ("", False),
            ("The viewer looks through the stack like looking through a window.", False),
            ("3D objects appear to float inside the glass volume with real", False),
            ("depth -- eyes naturally focus at the correct physical distance.", False),
            ("", False),
            ("=" * 64, False),
            ("LAYER TECHNOLOGIES", True),
            ("=" * 64, False),
            ("", False),
            ("T-OLED (Transparent Organic LED):", True),
            ("  - Transparent organic electrode matrix on glass substrate", False),
            ("  - Each pixel is independently addressable", False),
            ("  - Undriven pixels: fully transparent (~85% T)", False),
            ("  - Driven pixels: emit colored light at that depth plane", False),
            ("  - Response time: <1ms (instant on/off per frame)", False),
            ("", False),
            ("PDLC (Polymer-Dispersed Liquid Crystal):", True),
            ("  - Default state: fully transparent (clear glass)", False),
            ("  - Addressed pixels scatter light (become visible/opaque)", False),
            ("  - Used as depth barriers / parallax modulators", False),
            ("  - Switching speed: 2-5ms", False),
            ("", False),
            ("RE-Glass (Rare-Earth Doped):", True),
            ("  - Glass doped with rare-earth ions (invisible to naked eye)", False),
            ("  - Transparent until hit by NIR laser at specific wavelength", False),
            ("  - NIR excitation causes visible fluorescence at that voxel", False),
            ("  - Enables volumetric addressing without wiring each pixel", False),
            ("", False),
            ("=" * 64, False),
            ("FABRICATION BLUEPRINT", True),
            ("=" * 64, False),
            ("", False),
            ("STEP 1: Glass Substrate Preparation", True),
            ("  - Cut optically-flat borosilicate glass to panel size", False),
            ("  - Polish both surfaces to lambda/4 flatness", False),
            ("  - Clean in ultrasonic bath (acetone, IPA, DI water)", False),
            ("  - Apply AR coating via magnetron sputtering (both sides)", False),
            ("  - Target: <0.3% reflectance per surface (MgF2 + TiO2 stack)", False),
            ("", False),
            ("STEP 2: Active Layer Deposition", True),
            ("  - T-OLED: vacuum thermal evaporation of organic layers", False),
            ("    HIL/HTL/EML/ETL/EIL stack, transparent ITO cathode", False),
            ("  - PDLC: UV-cure polymer matrix with LC droplets, laminate", False),
            ("  - RE-Glass: melt-dope glass with Er3+/Yb3+ during formation", False),
            ("", False),
            ("STEP 3: Electrode Patterning", True),
            ("  - ITO transparent electrodes via photolithography", False),
            ("  - TFT backplane for active matrix addressing", False),
            ("  - Row/column drivers bonded via ACF (anisotropic film)", False),
            ("  - Resolution: match target panel (e.g., 3840x2160 per layer)", False),
            ("", False),
            ("STEP 4: Stack Assembly", True),
            ("  - Align layers using fiducial marks + micro-positioners", False),
            ("  - Bond with optical-grade UV-cure gel (n=1.47, matches glass)", False),
            ("  - Gel eliminates air gaps (no internal Fresnel reflections)", False),
            ("  - Spacing: 8-15mm front layers, 20-30mm rear layers", False),
            ("  - Cure under UV, then anneal at 60C for 24h", False),
            ("", False),
            ("STEP 5: Electronics Integration", True),
            ("  - Mount layer driver PCBs (one per 2-3 layers)", False),
            ("  - Connect via flex cables to each layer's edge contacts", False),
            ("  - Install GPU compute board (MPI depth-binning pipeline)", False),
            ("  - Wire DisplayPort inputs (DSC compressed, multi-port)", False),
            ("  - Install sync controller (phase-locks all layers to vsync)", False),
            ("", False),
            ("STEP 6: Enclosure & Thermal", True),
            ("  - CNC aluminium frame (bezel + rear housing)", False),
            ("  - Rear ventilation: heatsink fins + quiet PWM fans", False),
            ("  - Thermal budget: maintain <45C surface temperature", False),
            ("  - Cable routing through frame channels", False),
            ("", False),
            ("STEP 7: Calibration & Test", True),
            ("  - Per-layer uniformity calibration (9-point grid)", False),
            ("  - Depth registration: align all layers to <0.1mm", False),
            ("  - Crosstalk measurement: ghost images < -40dB", False),
            ("  - Luminance calibration: match brightness across layers", False),
            ("  - Color calibration: deltaE < 3 at all viewing angles", False),
            ("", False),
            ("=" * 64, False),
            ("GPU RENDERING PIPELINE", True),
            ("=" * 64, False),
            ("", False),
            ("The GPU converts 3D scene data to per-layer pixel assignments:", False),
            ("", False),
            ("  1. SCENE RENDER: Rasterize 3D scene to RGB + depth buffer", False),
            ("  2. DEPTH BINNING: Assign each pixel to nearest layer depth", False),
            ("  3. CORRELATION MATRIX: Build sparse M[pixel,layer] weights", False),
            ("  4. LAYER SCATTER: GPU distributes pixels to layer targets", False),
            ("  5. TEMPORAL REPROJECT: Reuse ~85% of prior frame data", False),
            ("  6. OCCLUSION RESOLVE: Fix transparency/overlap every 4 frames", False),
            ("  7. SYNC OUTPUT: Phase-locked delivery to all layers at vsync", False),
            ("", False),
            ("  Result: Each layer receives only its assigned pixels.", False),
            ("  Unassigned pixels remain transparent (layer stays clear).", False),
            ("", False),
            ("=" * 64, False),
            ("MATERIALS LIST (Desktop 32\" Reference)", True),
            ("=" * 64, False),
            ("", False),
            ("  - 10x borosilicate glass sheets (710x400mm, 1.1mm thick)", False),
            ("  - 10x AR coating (MgF2/TiO2 multilayer, both sides)", False),
            ("  - 6x T-OLED active layers (vacuum deposited)", False),
            ("  - 2x PDLC modulator films", False),
            ("  - 2x RE-doped glass sheets + NIR laser bars", False),
            ("  - 9x optical bonding gel (UV-cure, n=1.47)", False),
            ("  - 1x GPU board (24+ TFLOPS, 24GB HBM)", False),
            ("  - 4x layer driver PCBs", False),
            ("  - 1x sync controller + DisplayPort hub", False),
            ("  - 1x 600W PSU (80+ Platinum)", False),
            ("  - 1x CNC aluminium enclosure with heatsink", False),
            ("  - 2x 120mm PWM fans", False),
            ("", False),
            ("=" * 64, False),
            ("SPECIFICATIONS", True),
            ("=" * 64, False),
            ("", False),
        ]
        # add dynamic specs from current model
        mk = self.current_model_key()
        if mk:
            m = MODELS[mk]
            t_total, _ = model_stack_transmission(mk)
            pip = simulate_mpi_pipeline(mk)
            th = thermal_per_variant(mk)
            dq = model_depth_quality(mk)
            info_lines += [
                (f"  Current model: {m['name']}", False),
                (f"  Layers: {m['n_layers']} | Depth: {m['total_depth_m']*1000:.0f} mm", False),
                (f"  Resolution: {m['resolution_w']}x{m['resolution_h']} per layer", False),
                (f"  Target FPS: {m['target_fps']} | GPUs: {m['n_gpus']}", False),
                (f"  Stack transmission: {t_total*100:.1f}%", False),
                (f"  GPU pipeline: {pip['effective_ms']:.2f} ms / {pip['frame_budget_ms']:.1f} ms budget", False),
                (f"  Thermal: {th['total_w']:.0f} W total, {th['surface_temp_c']:.0f}C surface", False),
                (f"  Depth quality: {dq['quality_score']:.0f}/100", False),
                ("", False),
            ]

        info_lines += [
            ("=" * 64, False),
            ("", False),
            ("Press I to close | Scroll to navigate", True),
        ]

        # render lines with scroll
        margin_x, margin_y = 40, 30
        line_h = 16
        max_visible = (self.H - margin_y * 2) // line_h
        total_lines = len(info_lines)
        self.info_scroll = min(self.info_scroll, max(0, total_lines - max_visible))
        start = self.info_scroll
        end = min(start + max_visible, total_lines)

        for i, (text, bold) in enumerate(info_lines[start:end]):
            y = margin_y + i * line_h
            if bold:
                img = self.font_big.render(text, True, C_ACCENT)
            else:
                img = self.font.render(text, True, C_TEXT)
            self.screen.blit(img, (margin_x, y))

        # scroll indicator
        if total_lines > max_visible:
            bar_h = max(20, int(self.H * max_visible / total_lines))
            bar_y = int((self.H - bar_h) * start / max(1, total_lines - max_visible))
            pygame.draw.rect(self.screen, C_PANEL_HI, (self.W - 12, bar_y, 8, bar_h))

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self.handle_events()
            self.renderer.tick(dt)
            self.draw_bg()
            rect = pygame.Rect(0, 0, self.W, self.H)
            mpos = pygame.mouse.get_pos()
            self.renderer.render(self.screen, rect, self.show_labels,
                                 self.font, mouse_pos=mpos)
            self.draw_hud()
            self.draw_info()
            pygame.display.flip()
        pygame.quit()


# =============================================================================
# SECTION 9 -- SELFTEST & MAIN
# =============================================================================

def selftest():
    """Headless sanity check: build geometry, run physics, verify constraints."""
    print("GMDS selftest...")

    # test legacy build_display
    parts, depth = build_display()
    assert len(parts) > 10, f"Expected >10 parts, got {len(parts)}"
    assert abs(depth - DIMS["bezel_depth_mm"]*MM) < 1e-6

    t_total, per_layer = stack_transmission(DIMS["n_layers"])
    assert 0.0 < t_total < 1.0, f"Bad transmission: {t_total}"
    assert len(per_layer) == DIMS["n_layers"]

    pos = layer_z_positions(DIMS["n_layers"])
    assert len(pos) == DIMS["n_layers"]
    assert pos[0] == 0.0
    assert pos[-1] > 0.0

    th = thermal_analysis()
    assert th["total_w"] > 0

    bw = bandwidth_requirement()
    assert bw["total_gbps"] > 0

    rb = rendering_budget()
    assert rb["total_ms"] > 0

    ct = crosstalk_analysis()
    assert ct["ghost_db"] < 0

    lp, lz = build_layer_closeup()
    assert len(lp) > 0

    print(f"  Legacy: {len(parts)} parts, depth={depth*1000:.0f} mm")

    # test ALL model variants
    for key in MODELS:
        m = MODELS[key]
        vparts, vdepth = build_model_variant(key)
        assert len(vparts) >= m["n_layers"] + 2, f"{key}: too few parts"
        assert vdepth > 0

        vt, vpl = model_stack_transmission(key)
        assert 0.0 < vt < 1.0, f"{key}: bad T={vt}"
        assert len(vpl) == m["n_layers"]

        vrb = model_rendering_budget(key)
        assert vrb["total_ms"] > 0

        vcm = model_correlation_matrix_size(key)
        assert vcm["matrix_mb"] > 0
        assert vcm["nnz"] > 0

        vdq = model_depth_quality(key)
        assert 0 <= vdq["quality_score"] <= 100

        vm = model_mass_estimate(key)
        assert vm["total_kg"] > 0

        print(f"  {key}: {len(vparts)} parts, {m['n_layers']} layers, "
              f"T={vt*100:.1f}%, depth={m['total_depth_m']*1000:.0f} mm, "
              f"quality={vdq['quality_score']:.0f}/100, "
              f"corr matrix={vcm['matrix_mb']:.0f} MB")

    print(f"\n  Pipeline method: {PIPELINE['correlation_matrix_method']}")
    print(f"  Materials: {', '.join(MATERIALS.keys())}")

    # test ADVANCED physics
    print("\n  Advanced physics validation:")
    # Fresnel angular
    rs, rp, ravg = fresnel_angular(1.0, 1.5, 0.0)
    assert abs(ravg - fresnel_reflectance(1.0, 1.5)) < 1e-10, "Fresnel normal mismatch"
    rs45, rp45, r45 = fresnel_angular(1.0, 1.5, 45.0)
    assert r45 > ravg, "45-deg should reflect more than normal"
    # TIR
    rs_tir, rp_tir, r_tir = fresnel_angular(1.5, 1.0, 50.0)
    assert r_tir == 1.0, "Should be TIR"
    # Snell
    assert snell_angle(1.0, 1.5, 30.0) is not None
    assert snell_angle(1.5, 1.0, 50.0) is None  # TIR
    # Beer-Lambert
    assert 0 < beer_lambert(5.0, 0.001) < 1.0
    assert beer_lambert(0.0, 1.0) == 1.0  # no absorption
    # Luminance
    L = luminance_from_power(1.0, 0.1)
    assert L > 0
    print(f"    Fresnel/Snell/Beer-Lambert/Luminance: OK")

    # Pipeline simulation per variant
    for key in MODELS:
        pip = simulate_mpi_pipeline(key)
        assert pip["effective_ms"] > 0
        assert pip["meets_budget"], f"{key}: pipeline exceeds budget"
        dp = dp_link_budget(key)
        assert dp["raw_total_gbps"] > 0
        th = thermal_per_variant(key)
        assert th["total_w"] > 0
        sync = sync_timing_analysis(key)
        assert sync["frame_us"] > 0
        opt = full_optical_report(key)
        assert opt["t_total"] > 0
    print(f"    7-stage pipeline + DP + thermal + sync + optics: OK (all variants)")

    # Prototyping stages
    assert len(PROTOTYPE_STAGES) == 3
    for sk, sv in PROTOTYPE_STAGES.items():
        assert len(sv["bom"]) > 5
        assert sv["estimated_cost_usd"] > 0
        assert len(sv["risks"]) >= 2
    print(f"    Prototyping plan (3 stages): OK")

    # Risk register + measurement
    assert len(RISK_REGISTER) >= 6
    assert len(MEASUREMENT_METHODS) >= 8
    print(f"    Risk register ({len(RISK_REGISTER)} items) + Measurement ({len(MEASUREMENT_METHODS)} methods): OK")

    print("\nGMDS selftest PASSED -- all systems nominal.")
    return True


def benchmark():
    """Headless benchmark: build all variants, time physics + geometry."""
    import timeit
    print("GMDS Benchmark")
    print("=" * 60)

    # Geometry build times
    for key in MODELS:
        t = timeit.timeit(lambda: build_model_variant(key), number=10)
        print(f"  build_model_variant({key}): {t/10*1000:.1f} ms avg")

    # Physics computation times (first call + cached)
    # Clear caches
    for fn in [model_stack_transmission, model_rendering_budget, model_depth_quality,
               simulate_mpi_pipeline, dp_link_budget, thermal_per_variant,
               sync_timing_analysis, full_optical_report, model_layer_positions,
               model_correlation_matrix_size, model_mass_estimate, per_layer_power_model]:
        fn.cache_clear()

    for key in MODELS:
        t0 = timeit.timeit(lambda k=key: full_optical_report(k), number=1)
        # clear and time uncached
        full_optical_report.cache_clear()
        model_stack_transmission.cache_clear()
        model_layer_positions.cache_clear()
        t1 = timeit.timeit(lambda k=key: full_optical_report(k), number=1)
        # cached call
        t2 = timeit.timeit(lambda k=key: full_optical_report(k), number=100)
        print(f"  full_optical_report({key}): uncached={t1*1000:.2f} ms, "
              f"cached={t2/100*1000:.4f} ms ({t1/max(t2/100,1e-9):.0f}x speedup)")

    # Render benchmark (headless - no display)
    print("\n  Render benchmark (simulated 100 frames, no display):")
    for key in MODELS:
        parts, depth = build_model_variant(key)
        r = Renderer(parts, home_az=0.45, home_el=0.25, home_dist=0.5,
                     scale=0.5, center=(0, 0, depth/2))
        # create dummy surface
        surf = pygame.Surface((1280, 800))
        rect = pygame.Rect(0, 0, 1280, 800)
        t = timeit.timeit(lambda: r.render(surf, rect, False, None), number=100)
        fps_equiv = 100.0 / t
        print(f"    {key}: {t/100*1000:.1f} ms/frame ({fps_equiv:.0f} FPS equivalent)")

    print("\n" + "=" * 60)
    print("Benchmark complete.")


def main():
    parser = argparse.ArgumentParser(description="GMDS -- Volumetric 3D Display Digital Twin")
    parser.add_argument("--selftest", action="store_true", help="headless sanity check")
    parser.add_argument("--feasibility", action="store_true", help="full feasibility report")
    parser.add_argument("--export-obj", action="store_true", help="export OBJ mesh")
    parser.add_argument("--benchmark", action="store_true", help="headless performance benchmark")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.feasibility:
        print_feasibility()
        return
    if args.export_obj:
        export_obj()
        return
    if args.benchmark:
        pygame.init()
        benchmark()
        pygame.quit()
        return

    App().run()


if __name__ == "__main__":
    main()
