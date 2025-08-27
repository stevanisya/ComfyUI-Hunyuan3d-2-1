import trimesh
import pygltflib
import numpy as np
from PIL import Image
import base64
import os, glob, io
from PIL import Image
import numpy as np

def _pick_latest(dirpath: str, pattern: str):
    files = glob.glob(os.path.join(dirpath, pattern))
    return max(files, key=os.path.getmtime) if files else None

def _ensure_gray(path: str, value: int, size: int = 1024) -> str:
    """Create a flat grayscale image at 'path' if it doesn't exist."""
    if not path:
        return path
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(np.full((size, size), value, dtype=np.uint8)).save(path)
    return path

def _detect_temp_dir(obj_path: str, textures_dict: dict) -> str:
    """
    Try to infer a working temp directory from texture paths or next to the OBJ.
    Ensures the directory exists before returning it.
    """
    candidates = [
        os.path.dirname(textures_dict.get("metallic", "") or ""),
        os.path.dirname(textures_dict.get("roughness", "") or ""),
        os.path.join(os.path.dirname(obj_path) or ".", "temp"),
        os.path.join(os.getcwd(), "temp"),
    ]
    for d in candidates:
        if d and os.path.isdir(d):
            return d
    # fallback: create <obj_dir>/temp
    d = os.path.join(os.path.dirname(obj_path) or ".", "temp")
    os.makedirs(d, exist_ok=True)
    return d

def combine_metallic_roughness(metallic_path, roughness_path, output_path):
    """Pack Roughness→G, Metallic→B (R left as white for AO if absent)."""
    metallic_img  = Image.open(metallic_path).convert("L")
    roughness_img = Image.open(roughness_path).convert("L")

    if metallic_img.size != roughness_img.size:
        roughness_img = roughness_img.resize(metallic_img.size, Image.BICUBIC)

    width, height = metallic_img.size
    metallic_array  = np.array(metallic_img, dtype=np.uint8)
    roughness_array = np.array(roughness_img, dtype=np.uint8)

    combined_array = np.zeros((height, width, 3), dtype=np.uint8)
    combined_array[:, :, 0] = 255            # AO channel (R) = white when missing
    combined_array[:, :, 1] = roughness_array
    combined_array[:, :, 2] = metallic_array

    Image.fromarray(combined_array).save(output_path)
    return output_path


def create_glb_with_pbr_materials(obj_path, textures_dict, output_path):
    # 0) temp dir + safe output filename
    temp_dir = _detect_temp_dir(obj_path, textures_dict)
    os.makedirs(temp_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(obj_path))[0]
    if (not base) or base.startswith(".") or base.lower() == "obj":
        base = "asset"

    if (not output_path) or (os.path.basename(output_path) in ("", ".glb")):
        output_path = os.path.join(temp_dir, f"{base}.glb")

    # 1) Resolve metallic / roughness (.obj_metallic/.obj_roughness accepted)
    metal = textures_dict.get("metallic", "")
    rough = textures_dict.get("roughness", "")
    search_dir = os.path.dirname(metal or rough or obj_path) or "."

    if not os.path.exists(metal):
        cand = _pick_latest(search_dir, "*_metallic.*")
        if cand:
            metal = cand
    if not os.path.exists(rough):
        cand = _pick_latest(search_dir, "*_roughness.*")
        if cand:
            rough = cand

    metal = _ensure_gray(metal or os.path.join(search_dir, "_metallic.jpg"), 0)      # black = no metal
    rough = _ensure_gray(rough or os.path.join(search_dir, "_roughness.jpg"), 128)   # mid roughness
    textures_dict["metallic"]  = metal
    textures_dict["roughness"] = rough

    # 2) Albedo fallback (neutral mid-gray if missing)
    albedo = textures_dict.get("albedo", "")
    if not albedo or not os.path.exists(albedo):
        albedo = _ensure_gray(os.path.join(temp_dir, "_albedo.jpg"), 128)
    textures_dict["albedo"] = albedo

    # 3) Build combined metallicRoughness (G=roughness, B=metallic)
    mr_combined_path = os.path.join(temp_dir, "mr_combined.png")
    combine_metallic_roughness(metal, rough, mr_combined_path)
    textures_dict["metallicRoughness"] = mr_combined_path

    # 4) Export geometry to a temporary GLB, then load with pygltflib
    tmp_glb = os.path.join(temp_dir, f"{base}_tmp.glb")
    mesh = trimesh.load(obj_path, force="mesh")
    mesh.export(tmp_glb)  # writes real binary .glb with geometry
    gltf = pygltflib.GLTF2().load(tmp_glb)

    # 5) Attach textures/images to GLTF (deterministic order)
    def _guess_mime(p):
        ext = os.path.splitext(p)[1].lower()
        return {".jpg":"jpeg",".jpeg":"jpeg",".png":"png",".webp":"webp",".bmp":"bmp"}.get(ext,"png")

    def image_to_data_uri(p):
        with open(p, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/{_guess_mime(p)};base64,{data}"

    images, textures = [], []
    tex_index = {}

    def _add_tex(role, path):
        if not path or not os.path.exists(path):
            return
        images.append(pygltflib.Image(uri=image_to_data_uri(path)))
        textures.append(pygltflib.Texture(source=len(images) - 1))
        tex_index[role] = len(textures) - 1

    # Add in a known order
    _add_tex("albedo",            textures_dict.get("albedo"))
    _add_tex("metallicRoughness", textures_dict.get("metallicRoughness"))
    _add_tex("normal",            textures_dict.get("normal"))
    _add_tex("ao",                textures_dict.get("ao"))

    pbr = pygltflib.PbrMetallicRoughness(baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                                         metallicFactor=1.0, roughnessFactor=1.0)
    if "albedo" in tex_index:
        pbr.baseColorTexture = pygltflib.TextureInfo(index=tex_index["albedo"])
    if "metallicRoughness" in tex_index:
        pbr.metallicRoughnessTexture = pygltflib.TextureInfo(index=tex_index["metallicRoughness"])

    material = pygltflib.Material(name="PBR_Material", pbrMetallicRoughness=pbr)
    if "normal" in tex_index:
        material.normalTexture = pygltflib.NormalTextureInfo(index=tex_index["normal"])
    if "ao" in tex_index:
        material.occlusionTexture = pygltflib.OcclusionTextureInfo(index=tex_index["ao"])

    gltf.images = images
    gltf.textures = textures
    gltf.materials = [material]

    # Ensure first mesh uses our material
    if gltf.meshes:
        for prim in gltf.meshes[0].primitives:
            prim.material = 0

    # 6) Save as **binary** GLB
    gltf.save_binary(output_path)
    print(f"PBR GLB文件已保存: {output_path}")
    return output_path


