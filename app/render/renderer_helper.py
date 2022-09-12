import sys
import os

import kornia.morphology

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
import argparse
import sh

from scipy import ndimage as nd

from utils.transform import *
from utils.mtlparser import MTLParser
from floormesh.floormesh import  *

import lxml.etree as ET
from xml.dom import minidom
from guided_filter_pytorch.guided_filter import GuidedFilter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--house', type=str, default='0888')
    parser.add_argument('--floor', type=str, default='floor_01')
    parser.add_argument('--pano', type=str, default='pano_7')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--house', type=str, default='1434')
    parser.add_argument('--floor', type=str, default='floor_02')
    parser.add_argument('--pano', type=str, default='pano_33')
    # parser.add_argument('--split', type=str, default='val')
    # parser.add_argument('--house', type=str, default='1507')
    # parser.add_argument('--floor', type=str, default='floor_01')
    # parser.add_argument('--pano', type=str, default='pano_22')
    parser.add_argument('--sun_thresh_floor', type=int, default=0.3)
    parser.add_argument('--sun_thresh_wall', type=int, default=0.13)
    parser.add_argument('--log', type=str, default='out')
    parser.add_argument('--manual', action='store_false')
    parser.add_argument('--is_high_res', action='store_true')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--mask-samples', type=int, default=16)
    parser.add_argument('--sun-samples', type=int, default=16)
    parser.add_argument('--spec-samples', type=int, default=16)
    parser.add_argument('--blur-ksize', type=int, default=25)
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--mask-mesh-scale', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--layout-path', type=str, default='../data/zind/scenes/layout_merge')


    opt = parser.parse_args()
    return opt



def arch2pos(arch, camera_height=1.6, scale=4):
    B, H, W = arch.shape
    h = H // 2
    arch_floor = (arch == ARCH_FLOOR)[:,None].float()
    arch_wall = (arch == ARCH_WALL)[:,None].float()
    arch_floor_h = arch_floor[:,:,h:]
    arch_wall_h = arch_wall[:,:,h:]
    fline_wall_h = h - arch_wall_h.flip(2).argmax(dim=2, keepdim=True)
    fline_floor_h = arch_floor_h.argmax(dim=2, keepdim=True)
    fline_h = (fline_wall_h + fline_floor_h) / 2
    fline = h + fline_h

    H, W = H // scale, W // scale
    arch_floor = F.interpolate(arch_floor, (H, W), mode='area')
    arch_wall = F.interpolate(arch_wall, (H, W), mode='area')
    fline = F.interpolate(fline, (1, W), mode='area') / scale
    fline_lower = fline.floor().long() #B11W
    fline_upper = fline.ceil().long() # B11W
    fline_alpha = fline - fline_lower.float()

    uv = uvgrid(W, H, B)
    pix = uv2pix(uv)
    sph = pix2sph(pix)
    phi = sph[:,1:2]

    rho_floor = -camera_height / torch.sin(phi)
    sph_floor = torch.cat((sph, rho_floor), 1)
    pos_floor = sph2cart(sph_floor) # floor position
    pos_wall_xy_lower = torch.gather(pos_floor[:,:2], 2, fline_lower.expand(B, 2, H, W))
    pos_wall_xy_upper = torch.gather(pos_floor[:,:2], 2, fline_upper.expand(B, 2, H, W))
    pos_wall_xy = pos_wall_xy_upper * fline_alpha + pos_wall_xy_lower * (1 - fline_alpha)
    pos_wall_z = torch.linalg.norm(pos_wall_xy, dim=1, keepdim=True) * torch.tan(phi)
    pos_wall = torch.cat((pos_wall_xy, pos_wall_z), 1)
    pos = pos_floor * arch_floor + pos_wall * arch_wall
    return pos



def sun_floor2wall_grid(dirs, pos, extr=None, return_cart=False):
    """
        Directional Light warp grid
        T = S + k L, S: wall, T: floor
        dirs (L): N x 3, S->T direction
        pos (S): 1 x C x H x W
        ret: sampling grid: N x 2 x H x W
    """
    if extr is not None:
        pos = cam2wld(pos, extr)
    Tz = pos[:,2,-1,:].mean()
    L = dirs
    S = pos
    k = (Tz - S[:,2:]) / L[:,2:,None,None] # N x 1 x H x W
    T = S + k * L[:,:,None,None]
    if extr is not None:
        T = wld2cam(T, extr.expand(T.shape[0], -1))

    if return_cart:
        return T
    pix = cam2pix(T)
    return pix


def sun_wall2wall_grid(dirs, pos, extr=None, wpos=None):
    """
        Directional Light warp grid
        T = S + k L
        dirs (L): N x 3, S->T direction
        pos (S): 1 x C x H x W
        wpos: 2 x W
        ret: sampling grid:   N x 2 x H x W
             S validity: N x 1 x 1 x W
    """
    if extr is not None:
        pos = cam2wld(pos, extr)
    L = dirs
    S = pos
    if wpos is None:
        wpos = pos[0, :2, pos.shape[2]//2, :] # 2 x W
    wx = wpos[0]
    wy = wpos[1]
    Dx = wx[None,:] - wx[:,None] # W x W (S x T)
    Dy = wy[None,:] - wy[:,None] # W x W
    Lx = dirs[:,0]
    Ly = dirs[:,1]
    dot = (Lx[:,None,None] * Dx[None] + Ly[:,None,None] * Dy[None]) / torch.clamp((Dx**2 + Dy**2)**0.5, min=1e-10)[None] # N x W x W
    values, indices = dot.max(dim=2)      # N x W
    valid = (values > 0.99).float()[:, None, None, :] # N x 1 x 1 x W
    Sxy = wpos[None]                  # 1 x 2 x W
    Txy = wpos[:, indices].permute(1,0,2) # N x 2 x W
    k = torch.linalg.norm(Sxy - Txy, dim=1) # N x W
    T = S + k[:,None,None] * L[:,:,None,None]
    if extr is not None:
        T = wld2cam(T, extr.expand(T.shape[0], -1))
    pix = cam2pix(T)
    return pix, valid


def project_w2w(opt, suns, pos, rgb, mask=None):
    if opt.is_cuda:
        suns, pos, rgb = suns.cuda(), pos.cuda(), rgb.cuda()
    pix, valid = sun_wall2wall_grid(suns, pos)
    prgb = grid_sample(rgb, pix) * valid
    if mask is not None:
        if opt.is_cuda:
            mask = mask.cuda()
        pmask = grid_sample(mask, pix) * valid
        prgb, pmask = prgb.cpu(), pmask.cpu()
    else:
        return prgb.cpu()
    return prgb, pmask

def project_f2w(opt, suns, pos, rgb, mask=None):
    if opt.is_cuda:
        suns, pos, rgb = suns.cuda(), pos.cuda(), rgb.cuda()
    pix = sun_floor2wall_grid(suns, pos)
    prgb = grid_sample(rgb, pix)
    if mask is not None:
        if opt.is_cuda:
            mask = mask.cuda()
        pmask = grid_sample(mask, pix)
        prgb, pmask = prgb.cpu(), pmask.cpu()
    else:
        return prgb.cpu()
    return prgb, pmask


def project_w2f(suns, pos, rgb, mask=None):
    if opt.is_cuda:
        suns, pos, rgb = suns.cuda(), pos.cuda(), rgb.cuda()
    pix = sun_wall2floor_grid(suns, pos)
    prgb = grid_sample(rgb, pix)
    if mask is not None:
        if opt.is_cuda:
            mask = mask.cuda()
        pmask = grid_sample(mask, pix)
        prgb, pmask = prgb.cpu(), pmask.cpu()
    else:
        return prgb.cpu()
    return prgb, pmask


def project_f2w2f(opt, suns_ori, suns_new, pos, rgb, mask=None):
    if opt.is_cuda:
        suns_ori, suns_new, pos, rgb = suns_ori.cuda(), suns_new.cuda(), pos.cuda(), rgb.cuda()
    pos = sun_floor2wall_grid(suns_new, pos, return_cart=True)
    pix = sun_wall2floor_grid(-suns_ori, pos)
    prgb = grid_sample(rgb, pix)
    if mask is not None:
        if opt.is_cuda:
            mask = mask.cuda()
        pmask = grid_sample(mask, pix)
        prgb, pmask = prgb.cpu(), pmask.cpu()
    else:
        return prgb.cpu()
    return prgb, pmask


def parse_xml(xml, **kwargs):

    if xml.endswith('.xml'):
        scene = ET.parse(xml, ET.XMLParser(remove_blank_text=True)).getroot()
    else:
        scene = ET.fromstring(xml, ET.XMLParser(remove_blank_text=True))
    for k1, v1 in kwargs.items():
        for elem in scene.iter():
            for k2, v2 in elem.attrib.items():
                if v2 == '$' + k1:
                    elem.attrib[k2] = str(v1)
    return scene

def write_xml(scene):
    return minidom.parseString(ET.tostring(scene)).toprettyxml(indent='    ')[23:]

def build_energy_conservation():
    return ET.fromstring('<boolean name="ensureEnergyConservation" value="false"/>')

def build_arch(opt, split_id, house_id, floor_id, pano_id, mesh_id, bsdf_id, position=None, texture=None, roughness=0.03,
               mesh_dir=None, radiance=None, uvscale=None, ground_floor=False):
    mesh = ET.Element('shape')
    mesh.set('type', 'obj')
    if mesh_dir is None:
        mesh.append(parse_xml(f'<string name="filename" value="../data/zind/scenes/floormesh/{split_id}/{split_id}+{house_id}+{floor_id}+{pano_id}/{mesh_id}.obj"/>'))
    else:
        mesh.append(parse_xml(f'<string name="filename" value="{mesh_dir}.obj"/>'))

    to_world_path = f'../data/zind/scenes/floormesh/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/toworld_arch.xml'
    if os.path.exists(to_world_path):
        transform = parse_xml(to_world_path)
    else:
        if not ground_floor:
            transform = parse_xml(f'configs/transform/toworld.xml')
        else:
            transform =  parse_xml(f'configs/transform/toworld_ucsd.xml')


    if position is not None:
        transform.append(parse_xml(f'<translate x="{position[0]}" y="{position[1]}" z="{position[2]}" />'))
    mesh.append(transform)

    if bsdf_id == 'white':
        bsdf = parse_xml('configs/bsdfs/white.xml')
    elif bsdf_id == 'black':
        bsdf = parse_xml('configs/bsdfs/black.xml')
    elif bsdf_id == 'trans':
        bsdf = parse_xml('configs/bsdfs/trans.xml')
    elif bsdf_id == 'mirror':
        bsdf = parse_xml('configs/bsdfs/mirror.xml')
    elif bsdf_id == 'specular':
        bsdf = parse_xml('configs/bsdfs/roughmirror.xml', roughness=roughness)
    elif bsdf_id == 'mask':
        bsdf = parse_xml('configs/bsdfs/mask.xml', texture=texture)
    elif bsdf_id == 'difftrans':
        bsdf = parse_xml('configs/bsdfs/difftrans.xml', texture=texture)
    elif bsdf_id == 'area':
        bsdf = parse_xml('configs/emitters/area.xml')
    elif bsdf_id == 'occlusion_mask':
        bsdf = parse_xml('configs/bsdfs/occlusionmask.xml')
    elif bsdf_id == 'plastic':
        bsdf = parse_xml('configs/bsdfs/roughplastic.xml', roughness=roughness, texture=texture)
    elif bsdf_id == 'diffuse':
        bsdf = parse_xml('configs/bsdfs/diffuse.xml', texture=texture)
    elif bsdf_id == 'diffuse_uvscale':
        bsdf = parse_xml('configs/bsdfs/diffuse_uvscale.xml', texture=texture, uvscale=uvscale)
    if bsdf_id is not None:
        mesh.append(bsdf)
    return mesh

def build_furn(split_id, house_id, floor_id, pano_id, mesh_id, bsdf_id, position=None, texture=None, ground_floor=False, cube_only=False):

    mesh = ET.Element('shape')
    if not cube_only:
        mesh.set('type', 'obj')
        mesh.append(parse_xml(f'<string name="filename" value="{mesh_id}.obj"/>'))
        mesh.append(parse_xml(f'<float name="maxSmoothAngle" value="30"/>'))
    else:
        mesh.set('type', 'cube')

    if cube_only:

        # mesh.append(parse_xml(f'<point name="center" x="-0.05" y="0.04" z="-0.05"/>'))
        # mesh.append(parse_xml(f'<float name="radius" value="0.05"/>'))
        # mesh.append(parse_xml(f'<bsdf type="diffuse"/>'))
        transform = parse_xml(f'configs/transform/toworld_cube.xml')

    else:
        to_world_path = f'../data/zind/scenes/floormesh/{split_id}/{split_id}+{house_id}+{floor_id}+{pano_id}/toworld.xml'
        if os.path.exists(to_world_path):
            transform = parse_xml(to_world_path)
        else:
            if not ground_floor:
                transform = parse_xml(f'configs/transform/toworld.xml')
            else:
                transform =  parse_xml(f'configs/transform/toworld_ucsd.xml')

    if position is not None:
        transform.append(parse_xml(f'<translate x="{position[0]}" y="{position[1]}" z="{position[2]}" />'))
    mesh.append(transform)


    if bsdf_id == 'white':
        bsdf = parse_xml('configs/bsdfs/white.xml')
        mesh.append(bsdf)
    elif bsdf_id == 'black':
        bsdf = parse_xml('configs/bsdfs/black.xml')
        mesh.append(bsdf)
    elif bsdf_id == 'trans':
        bsdf = parse_xml('configs/bsdfs/trans.xml')
        mesh.append(bsdf)
    elif bsdf_id == 'mirror':
        bsdf = parse_xml('configs/bsdfs/mirror.xml')
        mesh.append(bsdf)
    elif bsdf_id == 'full':
        if not cube_only:
            xmls = MTLParser(f'{mesh_id}.mtl').save_mitsuba(None)
            for xml in xmls:
                bsdf = parse_xml(xml)
                mesh.append(bsdf)
        else:
            bsdf = parse_xml('configs/bsdfs/diffuse_lam.xml')
            mesh.append(bsdf)

    return mesh





def render_mitsuba(opt, scene, xml_fname='scene', render_fname='render'):
    xml = write_xml(scene)
    print_log(xml, fname=f'{opt.log}/{xml_fname}.xml', screen=False, mode='w')
    print_log(f'mitsuba {opt.log}/{xml_fname}.xml -o {opt.log}/{render_fname} -z', fname=f'{opt.log}/commands.sh', screen=False, mode='w')
    sh.bash(f'{opt.log}/commands.sh')


def add_furns(opt, scene, bsdf_id, select_name=None, ground_floor=False):
    if opt.cube_only:
        scene.append(build_furn(opt.split, opt.house, opt.floor, opt.pano, '', bsdf_id, ground_floor=ground_floor, cube_only=True))

    else:
        if opt.manual:
            data_path = f'../data/zind/manualfill/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/'
        else:
            data_path = f'../data/zind/furnmesh/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/'
        if select_name is None:
            for obj in Path(data_path).glob('*.obj'):
                mesh_id = obj.name[:-4]
                scene.append(build_furn(opt.split, opt.house, opt.floor, opt.pano, str(obj)[:-4], bsdf_id, ground_floor=ground_floor))
        else:
            for obj in Path(data_path).glob('*.obj'):
                if select_name in str(obj):
                    scene.append(build_furn(opt.split, opt.house, opt.floor, opt.pano, str(obj)[:-4], bsdf_id, ground_floor=ground_floor))

def render_sunlight(opt, sun, sunlight_t, furn=True, maxd=None, floor_bsdf='white', furn_bsdf='full', samples=16):
    save_hdr(f'{opt.log}/sunmask.exr', 1 - sunlight_t.flip(2) / 255)
    save_im(f'{opt.log}/sunmask.png', 1 - sunlight_t.flip(2) / 255)
    scene = parse_xml('configs/scenes/scene.xml')
    scene.append(parse_xml('configs/sensors/hdr.xml', height=opt.height, width=opt.height * 2, samples=samples))
    if maxd is None:
        scene.append(parse_xml('configs/integrators/volpath.xml'))
    else:
        scene.append(parse_xml('configs/integrators/volpath_depth.xml', maxd=maxd))

    scene.append(build_arch(opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', floor_bsdf))
    scene.append(build_arch(opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'mask', position=-sun[0] * 255, texture=f'{opt.log}/sunmask.exr'))
    scene.append(parse_xml('configs/emitters/directional.xml', sundirx=sun[0,0].item(), sundiry=sun[0,1].item(), sundirz=sun[0,2].item()))

    if furn:
        add_furns(opt, scene, furn_bsdf)

    render_mitsuba(scene)
    render = torch.from_numpy(cv2.imread(f'{opt.log}/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]
    render = render * math.pi / torch.sin(-torch.atan(sun[0,2])) * 255
    return render


def render_sunlight_old(opt, sun, sunlight_t, furn=True, maxd=None, floor_bsdf='white', furn_bsdf='full', samples=16):
    save_hdr(f'{opt.log}/sunmask.exr', 1 - sunlight_t.flip(2) / 255)
    save_im(f'{opt.log}/sunmask.png', 1 - sunlight_t.flip(2) / 255)
    scene = parse_xml('configs/scenes/scene.xml')
    scene.append(parse_xml('configs/sensors/hdr.xml', height=opt.height, width=opt.height * 2, samples=samples))
    if maxd is None:
        scene.append(parse_xml('configs/integrators/volpath.xml'))
    else:
        scene.append(parse_xml('configs/integrators/volpath_depth.xml', maxd=maxd))

    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', floor_bsdf))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'mask', position=-sun[0] * 255, texture=f'{opt.log}/sunmask.exr'))
    scene.append(parse_xml('configs/emitters/directional.xml', sundirx=sun[0,0].item(), sundiry=sun[0,1].item(), sundirz=sun[0,2].item()))

    if furn:
        add_furns(opt, scene, furn_bsdf)

    render_mitsuba(opt, scene)
    render = torch.from_numpy(cv2.imread(f'{opt.log}/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]
    render = render * math.pi / torch.sin(-torch.atan(sun[0,2])) * 255
    return render

def render_sunlight(opt, sun, sunlight_t, bright_wall_mesh_dir, dark_wall_mesh_dir, bright_rt_wall_mesh_dir, dark_rt_wall_mesh_dir, furn=True, maxd=None,
                    floor_bsdf='white', wall_bsdf='white', furn_bsdf='full', samples=16, normal=None, factor=20):
    save_hdr(f'{opt.log}/sunmask.exr', 1 - sunlight_t.flip(2) / factor)
    save_im(f'{opt.log}/sunmask.png',  sunlight_t.flip(2) / factor)
    scene = parse_xml('configs/scenes/scene.xml')
    scene.append(parse_xml('configs/sensors/hdr.xml', height=opt.height, width=opt.height * 2, samples=samples))
    if maxd is None:
        scene.append(parse_xml('configs/integrators/volpath.xml'))
    else:
        scene.append(parse_xml('configs/integrators/volpath_depth.xml', maxd=maxd))


    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', floor_bsdf))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_ceiling', 'occlusion_mask'))
    scene.append(build_arch(opt, None, None, None, None, None, wall_bsdf, mesh_dir=dark_wall_mesh_dir))

    scene.append(build_arch(opt, None, None, None, None, None, 'mask', mesh_dir= bright_wall_mesh_dir,position=np.array([0,0,0]), texture=f'{opt.log}/sunmask.exr'))
    scene.append(build_arch(opt, None, None, None, None, None, 'occlusion_mask', mesh_dir=dark_rt_wall_mesh_dir))
    scene.append(build_arch(opt, None, None, None, None, None, 'mask', mesh_dir= bright_rt_wall_mesh_dir,position=np.array([0,0,0]), texture=f'{opt.log}/sunmask.exr'))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'mask_ceiling', 'occlusion_mask'))

    # scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_wall', 'mask', position=np.array([0,0,0]), texture=f'{opt.log}/sunmask.exr'))
    scene.append(parse_xml('configs/emitters/directional.xml', sundirx=sun[0,0].item(), sundiry=sun[0,1].item(), sundirz=sun[0,2].item()))

    if furn:
        add_furns(opt, scene, furn_bsdf)

    render_mitsuba(opt, scene)
    render = torch.from_numpy(cv2.imread(f'{opt.log}/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]
    print(render.max())
    if normal is not None:
        sinz = torch.sin(-torch.atan(sun[0,2]))
        cosz = torch.cos(-torch.atan(sun[0,2]))
        sundir = torch.Tensor([-sun[0,0] * cosz, -sun[0,1] * cosz, sinz])[None, :, None, None]

        indir = torch.sum(normal * sundir, dim=1, keepdim=True)

        render = render * math.pi * factor * factor / indir
    else:
        render = render * math.pi * factor * factor
    # math.cos(azimuth * math.pi / 180), math.sin(azimuth * math.pi / 180), -math.tan(elevation * math.pi / 180)

    # / torch.sin(-torch.atan(sun[0,2]))

    return render





def render_plastic_sun(opt, roughness, sun, sunlight_t, bright_wall_mesh_dir, dark_wall_mesh_dir, bright_rt_wall_mesh_dir, dark_rt_wall_mesh_dir, maxd=None, samples=16, floor_tex_dir=None, ratio=1, normal=None, factor=20):    # sunlight_t = sunlight_t * math.pi / torch.sin(-torch.atan(sun[0,2]))
    save_hdr(f'{opt.log}/sunmask.exr', 1 - sunlight_t.flip(2) * ratio/ factor)
    save_im(f'{opt.log}/sunmask.png',  sunlight_t.flip(2) *ratio/ factor)

    # plt.imshow((1 - sunlight_t.flip(2) / 255)[0,0])
    scene = parse_xml('configs/scenes/scene.xml')
    # scene.append(parse_xml('configs/sensors/tonemap.xml', height=opt.height, width=opt.height * 2, samples=samples))
    scene.append(parse_xml('configs/sensors/hdr.xml', height=opt.height, width=opt.height * 2, samples=samples))

    if maxd is None:
        scene.append(parse_xml('configs/integrators/volpath.xml'))
    else:
        scene.append(parse_xml('configs/integrators/volpath_depth.xml', maxd=maxd))

    if floor_tex_dir is None:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'plastic', texture=f'{opt.cache}/tex_floor_ambi.png', roughness=roughness))
    else:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'plastic', texture=floor_tex_dir, roughness=roughness))


    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_ceiling', 'occlusion_mask'))
    scene.append(build_arch(opt, None, None, None, None, None, 'occlusion_mask', mesh_dir=dark_wall_mesh_dir ))
    scene.append(build_arch(opt, None, None, None, None, None, 'mask', mesh_dir= bright_wall_mesh_dir,position=np.array([0,0,0]), texture=f'{opt.log}/sunmask.exr'))
    scene.append(build_arch(opt, None, None, None, None, None, 'occlusion_mask', mesh_dir=dark_rt_wall_mesh_dir))
    scene.append(build_arch(opt, None, None, None, None, None, 'mask', mesh_dir= bright_rt_wall_mesh_dir,position=np.array([0,0,0]), texture=f'{opt.log}/sunmask.exr'))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'mask_ceiling', 'occlusion_mask'))

    # scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_wall', 'mask', position=np.array([0,0,0]), texture=f'{opt.log}/sunmask.exr'))
    scene.append(parse_xml('configs/emitters/directional.xml', sundirx=sun[0,0].item(), sundiry=sun[0,1].item(), sundirz=sun[0,2].item()))

    render_mitsuba(opt, scene)
    # render = read_im(f'{opt.log}/render.png')
    # render = render * math.pi / torch.sin(-torch.atan(sun[0,2])) * 255
    render = torch.from_numpy(cv2.imread(f'{opt.log}/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]

    sinz = torch.sin(-torch.atan(sun[0,2]))
    cosz = torch.cos(-torch.atan(sun[0,2]))
    sundir = torch.Tensor([-sun[0,0] * cosz, -sun[0,1] * cosz, sinz])[None, :, None, None]

    indir = torch.sum(normal * sundir, dim=1, keepdim=True)

    render = render * math.pi * factor * factor / indir

    return render


def render_plastic(opt, roughness, texture, radiance, samples):
    scene = parse_xml('configs/scenes/scene.xml')
    scene.append(parse_xml('configs/sensors/hdr.xml', height=opt.height, width=opt.height * 2, samples=samples))
    scene.append(parse_xml('configs/integrators/path.xml'))

    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'plastic', texture=texture,
                            roughness=roughness))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_wall', 'difftrans',
                            texture=f'../data/zind/scenes/floormesh/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/tex_wall.exr'))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_ceiling', 'difftrans',
                            texture=f'../data/zind/scenes/floormesh/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/tex_ceiling.exr'))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'light_wall', 'area', radiance=radiance))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'light_ceiling', 'area', radiance=radiance))

    render_mitsuba(opt, scene)
    render = torch.from_numpy(cv2.imread(f'{opt.log}/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]
    return render

def render_diffuse_sun(opt, sun, sunlight_t, bright_wall_mesh_dir, dark_wall_mesh_dir, bright_rt_wall_mesh_dir, dark_rt_wall_mesh_dir, maxd=None, samples=16, ratio=1, normal=None, factor=20, texture=None):
    save_hdr(f'{opt.log}/sunmask.exr', 1 - sunlight_t.flip(2) * ratio / factor)
    save_im(f'{opt.log}/sunmask.png',  sunlight_t.flip(2) * ratio / factor)

    scene = parse_xml('configs/scenes/scene.xml')
    scene.append(parse_xml('configs/sensors/hdr.xml', height=opt.height, width=opt.height * 2, samples=samples))

    if maxd is None:
        scene.append(parse_xml('configs/integrators/volpath.xml'))
    else:
        scene.append(parse_xml('configs/integrators/volpath_depth.xml', maxd=maxd))


    scene.append(build_arch(opt, None, None, None, None, None, 'black', mesh_dir=dark_wall_mesh_dir, texture=f'{dark_wall_mesh_dir}.png'))
    scene.append(build_arch(opt, None, None, None, None, None, 'black', mesh_dir=dark_rt_wall_mesh_dir))

    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'diffuse', texture=texture))

    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_ceiling', 'occlusion_mask'))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'mask_ceiling', 'occlusion_mask'))

    scene.append(build_arch(opt, None, None, None, None, None, 'mask', position=np.array([0,0,0]), texture=f'{opt.log}/sunmask.exr', mesh_dir=bright_wall_mesh_dir))
    scene.append(build_arch(opt, None, None, None, None, None, 'mask', mesh_dir=bright_rt_wall_mesh_dir, position=np.array([0, 0, 0]), texture=f'{opt.log}/sunmask.exr'))


    scene.append(parse_xml('configs/emitters/directional.xml', sundirx=sun[0,0].item(), sundiry=sun[0,1].item(), sundirz=sun[0,2].item()))

    render_mitsuba(opt, scene)
    render = torch.from_numpy(cv2.imread(f'{opt.log}/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]

    sinz = torch.sin(-torch.atan(sun[0,2]))
    cosz = torch.cos(-torch.atan(sun[0,2]))
    sundir = torch.Tensor([-sun[0,0] * cosz, -sun[0,1] * cosz, sinz])[None, :, None, None]

    indir = torch.sum(normal * sundir, dim=1, keepdim=True)

    render = render * math.pi * factor * factor / indir

    return render



def render_specular(opt, mirror_spec_i, furn=True, roughness=0.03, samples=16):
    save_hdr(f'{opt.log}/specmap.exr', mirror_spec_i)
    scene = parse_xml('configs/scenes/scene.xml')
    scene.append(parse_xml('configs/sensors/hdr.xml', height=opt.height, width=opt.height * 2, samples=samples))
    scene.append(parse_xml('configs/integrators/direct.xml'))

    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'specular', roughness=roughness))
    scene.append(parse_xml('configs/emitters/envmap.xml', texture=f'{opt.log}/specmap.exr'))
    if furn:
        add_furns(opt, scene, 'black')

    render_mitsuba(opt, scene)
    render = torch.from_numpy(cv2.imread(f'{opt.log}/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]
    # render[:,:,:render.shape[2]//2] = 0
    return render



def render_ambient(opt, furn=True, white_floor=True, white_wall=False, wall_tex_dir=None, floor_tex_dir=None, sample_scale=1):
    scene = parse_xml('configs/scenes/scene.xml')
    scene.append(parse_xml('configs/sensors/hdr.xml', height=opt.height, width=opt.height * 2, samples=opt.samples * sample_scale))
    scene.append(parse_xml('configs/integrators/path.xml'))


    if white_floor:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'white'))
    else:
        if floor_tex_dir is None:
            scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'difftrans', texture=f'../data/zind/scenes/floormesh/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/tex_floor.png'))
        else:
            scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'difftrans', texture=floor_tex_dir))

    if white_wall:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_wall', 'white'))
    else:
        if wall_tex_dir is None:
            if opt.blur_ksize > 0:
                scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_wall', 'difftrans', texture=f'{opt.cache}/tex_wall_blur.exr'))
            else:
                scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_wall', 'difftrans', texture=f'../data/zind/scenes/floormesh/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/tex_wall.exr'))
        else:
            scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_wall', 'difftrans', texture=wall_tex_dir))

    if opt.blur_ksize > 0:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_ceiling', 'difftrans', texture=f'{opt.cache}/tex_ceiling_blur.exr'))
    else:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_ceiling', 'difftrans', texture=f'../data/zind/scenes/floormesh/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/tex_ceiling.exr'))
    if not white_floor:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, f'light_floor', 'area'))
    if not white_wall:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, f'light_wall', 'area'))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, f'light_ceiling', 'area'))

    # scene.append(parse_xml('configs/emitters/envmap.xml', texture=f'../data/zind/scenes/light/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}_hdr.exr'))

    if furn:
        add_furns(opt, scene, 'full')

    render_mitsuba(opt, scene)
    render = torch.from_numpy(cv2.imread(f'{opt.log}/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]
    return render




def render_ambient_plastic(opt, furn=True, white_floor=True, white_wall=False, wall_tex_dir=None, floor_tex_dir=None, sample_scale=1, roughness=0.03):
    scene = parse_xml('configs/scenes/scene.xml')
    scene.append(parse_xml('configs/sensors/hdr.xml', height=opt.height, width=opt.height * 2, samples=opt.samples * sample_scale))
    scene.append(parse_xml('configs/integrators/path.xml'))


    if white_floor:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'white'))
    else:
        if floor_tex_dir is None:
            scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'plastic', roughness=roughness, texture=f'../data/zind/scenes/floormesh/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/tex_floor.png'))
        else:
            scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'plastic', roughness=roughness, texture=floor_tex_dir))

    if white_wall:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_wall', 'white'))
    else:
        if wall_tex_dir is None:
            if opt.blur_ksize > 0:
                scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_wall', 'difftrans', texture=f'{opt.cache}/tex_wall_blur.exr'))
            else:
                scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_wall', 'difftrans', texture=f'../data/zind/scenes/floormesh/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/tex_wall.exr'))
        else:
            scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_wall', 'difftrans', texture=wall_tex_dir))

    if opt.blur_ksize > 0:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_ceiling', 'difftrans', texture=f'{opt.cache}/tex_ceiling_blur.exr'))
    else:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_ceiling', 'difftrans', texture=f'../data/zind/scenes/floormesh/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/tex_ceiling.exr'))
    if not white_floor:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, f'light_floor', 'area'))
    if not white_wall:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, f'light_wall', 'area'))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, f'light_ceiling', 'area'))

    # scene.append(parse_xml('configs/emitters/envmap.xml', texture=f'../data/zind/scenes/light/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}_hdr.exr'))

    if furn:
        add_furns(opt, scene, 'full')

    render_mitsuba(opt, scene)
    render = torch.from_numpy(cv2.imread(f'{opt.log}/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]
    return render



def render_ambient_ucsd(opt, furn=True, envmap=None, furn_name=None, white_floor=True, white_wall=False, wall_tex_dir=None, floor_tex_dir=None, sample_scale=1):
    scene = parse_xml('configs/scenes/scene.xml')
    scene.append(parse_xml('configs/sensors/hdr_ucsd.xml', height=opt.height, width=opt.height * 2, samples=opt.samples * sample_scale))
    scene.append(parse_xml('configs/integrators/path.xml'))
    save_hdr(f'{opt.cache}/envmap.exr', envmap)

    scene.append(parse_xml('configs/emitters/envmap_4.xml', texture=f'{opt.cache}/envmap.exr'))

    if white_floor:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'white', ground_floor=True))
    else:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'difftrans',
                                texture=f'../data/zind/scenes/floormesh/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/tex_floor.png', ground_floor=True))

    # scene.append(parse_xml('configs/emitters/envmap.xml', texture=f'../data/zind/scenes/light/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}_hdr.exr'))

    if furn:
        add_furns(opt, scene, 'full', furn_name, ground_floor=True)

    render_mitsuba(opt, scene)
    render = torch.from_numpy(cv2.imread(f'{opt.log}/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]
    return render



def render_diffuse(opt, roughness, furn=True, white_floor=True, white_wall=False, wall_tex_dir=None, floor_tex_dir=None, sample_scale=8):
    scene = parse_xml('configs/scenes/scene.xml')
    scene.append(parse_xml('configs/sensors/hdr.xml', height=opt.height, width=opt.height * 2, samples=opt.samples * sample_scale))
    scene.append(parse_xml('configs/integrators/path.xml'))


    if white_floor:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'white'))
    else:
        if floor_tex_dir is None:
            scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'difftrans', texture=f'../data/zind/scenes/floormesh/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/tex_floor.png'))
        else:
            scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'difftrans', texture=floor_tex_dir))

    if white_wall:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_wall', 'white'))
    else:
        if wall_tex_dir is None:
            scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_wall', 'difftrans', texture=f'../data/zind/scenes/floormesh/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/tex_wall.exr'))
        else:
            scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_wall', 'difftrans', texture=wall_tex_dir))

    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_ceiling', 'difftrans', texture=f'../data/zind/scenes/floormesh/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/tex_ceiling.exr'))
    if not white_floor:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, f'light_floor', 'area'))
    if not white_wall:
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, f'light_wall', 'area'))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, f'light_ceiling', 'area'))

    # scene.append(parse_xml('configs/emitters/envmap.xml', texture=f'../data/zind/scenes/light/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}_hdr.exr'))

    if furn:
        add_furns(opt, scene, 'full')

    render_mitsuba(opt, scene)
    render = torch.from_numpy(cv2.imread(f'{opt.log}/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]
    return render



def render_floormask(opt, furn=False):
    scene = parse_xml('configs/scenes/scene.xml')
    scene.append(parse_xml('configs/sensors/hdr.xml', height=opt.height , width=opt.height * 2 , samples=opt.mask_samples))
    scene.append(parse_xml('configs/integrators/albedo.xml'))

    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'white'))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_wall', 'trans'))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_ceiling', 'trans'))


    render_mitsuba(opt, scene)
    render = torch.from_numpy(cv2.imread(f'{opt.log}/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]
    return render


def render_wallmask(opt, furn=False):
    scene = parse_xml('configs/scenes/scene.xml')
    scene.append(parse_xml('configs/sensors/hdr.xml', height=opt.height , width=opt.height * 2 , samples=opt.mask_samples))
    scene.append(parse_xml('configs/integrators/albedo.xml'))

    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_floor', 'trans'))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_wall', 'white'))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_ceiling', 'trans'))



    render_mitsuba(opt, scene)
    render = torch.from_numpy(cv2.imread(f'{opt.log}/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]
    return render


def render_furnmask(opt, furn=True):
    scene = parse_xml('configs/scenes/scene.xml')
    scene.append(parse_xml('configs/sensors/hdr.xml', height=opt.height, width=opt.height * 2, samples=opt.mask_samples))
    scene.append(parse_xml('configs/integrators/albedo.xml'))

    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'trans'))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_wall', 'trans'))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_ceiling', 'trans'))

    if furn:
        add_furns(opt, scene, 'white')

    render_mitsuba(opt, scene)
    render = torch.from_numpy(cv2.imread(f'{opt.log}/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]
    return render


# def render_persp(opt, furn=True):
#     scene = parse_xml('configs/scenes/scene.xml')
#     scene.append(parse_xml('configs/sensors/ldr.xml', height=opt.height, width=opt.height * 2, samples=opt.mask_samples))
#     scene.append(parse_xml('configs/integrators/albedo.xml'))
#
#     scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'trans'))
#     scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_wall', 'trans'))
#     scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_ceiling', 'trans'))
#
#     if furn:
#         add_furns(opt, scene, 'white')
#
#     render_mitsuba(opt, scene)
#     render = torch.from_numpy(cv2.imread(f'{opt.log}/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]
#     return render





def render_furnmask_sep(opt, furn=True):
    if opt.manual:
        data_path = f'../data/zind/manualfill/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/'
    else:
        data_path = f'../data/zind/furnmesh/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/'

    mesh_ids = []
    renders = []
    # render furn mask separately
    i = 0
    for obj in Path(data_path).glob('*.obj'):

        i +=1
        scene = parse_xml('configs/scenes/scene.xml')
        scene.append(parse_xml('configs/sensors/hdr.xml', height=opt.height, width=opt.height * 2, samples=opt.mask_samples))
        scene.append(parse_xml('configs/integrators/albedo.xml'))

        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'tex_floor', 'trans', ground_floor=True))
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_wall', 'trans', ground_floor=True))
        scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_ceiling', 'trans', ground_floor=True))

        mesh_id = obj.name[:-4]
        scene.append(build_furn(opt.split, opt.house, opt.floor, opt.pano, str(obj)[:-4], 'white', ground_floor=True))

        render_mitsuba(opt, scene)
        render = torch.from_numpy(cv2.imread(f'{opt.log}/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]

        renders.append(render)
        mesh_ids.append(mesh_id)

    renders = torch.cat(renders, dim=0)
    print(renders.sum())
    return renders, mesh_ids


def render_normal(opt, furn=False):
    scene = parse_xml('configs/scenes/scene.xml')
    scene.append(parse_xml('configs/sensors/hdr.xml', height=opt.height, width=opt.height * 2, samples=opt.mask_samples))
    scene.append(parse_xml('configs/integrators/shadingnormal.xml'))

    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_wall', None ))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_floor', None ))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_ceiling', None))

    if furn:
        add_furns(opt, scene, 'white')

    render_mitsuba(opt, scene)
    render = torch.from_numpy(cv2.imread(f'{opt.log}/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]
    return render




def render_pos(opt, furn=False, scale=4):
    scene = parse_xml('configs/scenes/scene.xml')
    scene.append(parse_xml('configs/sensors/hdr.xml', height=opt.height, width=opt.height*2 , samples=opt.mask_samples))
    scene.append(parse_xml('configs/integrators/pos.xml'))

    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_wall', None ))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_floor', None ))
    scene.append(build_arch(opt, opt.split, opt.house, opt.floor, opt.pano, 'geo_ceiling', None))

    if furn:
        add_furns(opt, scene, 'white')

    render_mitsuba(opt, scene)
    render = torch.from_numpy(cv2.imread(f'{opt.log}/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]
    # render = F.interpolate(render, (opt.height, opt.height*2), mode='area')

    return render

def blur_tex(input_fname, output_fname, ksize):
    im = cv2.imread(input_fname, -1)
    H, W = im.shape[:2]
    N = W // H
    for i in range(N):
        crop = im[:, H*i:H*i+H]
        crop = cv2.GaussianBlur(crop, (ksize, ksize), 0)
        im[:, H*i:H*i+H] = crop
    cv2.imwrite(output_fname, im)


def get_sundir_vector(self, record_id):
    record = record_id.split(' ')
    split_id, house_id, floor_id, pano_id = record
    f = open(self.template['sundir'].format(split=split_id, house=house_id, floor=floor_id, pano=pano_id), 'r')
    lines = f.readlines()
    f.close()
    score, azimuth, elevation = lines[0].split(' ')
    score, azimuth, elevation = float(score), float(azimuth), float(elevation)
    vec = torch.Tensor([score, azimuth, elevation, math.cos(azimuth * math.pi / 180), math.sin(azimuth * math.pi / 180),
                        -math.tan(elevation * math.pi / 180)])
    return vec


def guided_filter(rgb, sem, thresh=0.55, r=21):
    hr_x_gray = kornia.rgb_to_grayscale(rgb)
    init_hr_y = kornia.gaussian_blur2d(sem, (15, 15), (1.5, 1.5))

    sem_filter = GuidedFilter(r)(hr_x_gray, init_hr_y)

    sem_filter = sem_filter > thresh
    sem_filter, _ = remove_small_island(sem_filter)
    sem_filter = morpho_close_pano(sem_filter, 7)
    return sem_filter.float()

def project_sun_forward(opt, sun, sun_new, sunlight_i, ambi_i, pos, sem_floor_i, sem_wall_i, use_main_color=True, is_backward=False, return_mask=False, sun_thresh_floor=0, sun_thresh_wall=0):
    # isun_i = diff_i / torch.clamp(ambi_i, 0.001) - 1
    # sunlight_i = isun_i

    # vis_torch_im(sunlight_i)



    sunlight_i_floor = sunlight_i * sem_floor_i
    sunlight_i_wall = sunlight_i * sem_wall_i
    #
    # vis_torch_im(sunlight_i_floor)
    # vis_torch_im(sunlight_i_wall)
    sem_wall_i_hr = render_wallmask(opt)

    pano_sun_f2w = project_f2w(opt, sun, pos, sunlight_i_floor)

    pano_sun_f2w = pano_sun_f2w * sem_wall_i_hr
    pano_sun_w2w = project_w2w(opt, sun, pos, sunlight_i_wall)
    pano_sun_w2w = pano_sun_w2w * sem_wall_i_hr

    pano_sun_f2w, pano_sun_f2w_valid_mask = remove_small_island(pano_sun_f2w, sun_thresh_floor)
    pano_sun_w2w, pano_sun_w2w_valid_mask = remove_small_island(pano_sun_w2w, sun_thresh_wall)
    # vis_torch_im(pano_sun_f2w)
    # vis_torch_im(pano_sun_w2w)
    # pano_sun_div = pano_sun_f2w + pano_sun_w2w
    pano_sun_div_valid_mask = (pano_sun_f2w_valid_mask + pano_sun_w2w_valid_mask) > 0
    # vis_torch_im(pano_sun_f2w )
    # vis_torch_im(pano_sun_w2w)

    if use_main_color:
        main_color_floor = get_dominant_color(torch2np_im(pano_sun_f2w), torch2np_im(pano_sun_f2w_valid_mask))
        main_color_wall = get_dominant_color(torch2np_im(pano_sun_w2w), torch2np_im(pano_sun_w2w_valid_mask))
        pano_sun_div_floor = pano_sun_div_valid_mask.float() * torch.Tensor(main_color_floor)[None, :, None, None]
        pano_sun_div_wall = pano_sun_div_valid_mask.float() * torch.Tensor(main_color_wall)[None, :, None, None]
        # vis_torch_im(pano_sun_div_wall)
        # vis_torch_im(pano_sun_div_floor)

    else:
        pano_sun_div_floor = pano_sun_f2w
        pano_sun_div_wall = pano_sun_w2w

    if is_backward:
        pano_sun_floor = project_w2f(-sun_new, pos, pano_sun_div_floor)
        pano_sun_floor = pano_sun_floor * sem_floor_i
        pano_sun_wall = project_w2w(-sun_new, pos, pano_sun_div_wall)
        pano_sun_wall = pano_sun_wall * sem_wall_i

        pano_sun_floor, _ = remove_small_island(pano_sun_floor)
        pano_sun_wall, _ = remove_small_island(pano_sun_wall)

        pano_ratio = pano_sun_floor + pano_sun_wall
        # vis_torch_im(pano_ratio)

        pano = ambi_i * pano_ratio
    else:
        if use_main_color:
            pano = pano_sun_div_floor
        else:
            pano = pano_sun_div_floor + pano_sun_div_wall
    pano = pano * sem_wall_i_hr
    # vis_torch_im(pano)

    if return_mask:
        return pano, pano_sun_f2w_valid_mask, pano_sun_w2w_valid_mask
    else:
        return pano

def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes.int()



def fit_bbox(mask, thresh=0):
    mask = mask > thresh
    if len(mask.shape) == 4:
        mask = mask[:, 0, ...]
    bboxs = masks_to_boxes(mask)
    bbox_mask = torch.zeros_like(mask)
    for idx in range(len(bboxs)):
        bbox = bboxs[idx]
        bbox_mask[0, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

    bbox_mask = bbox_mask[:, None, ...]
    return bbox_mask

def _fill_im(src_img, src_mask, dst_mask, mode, min_valid_ratio, src_mask_close=None):
    # src_img = src_img * src_mask
    # vis_torch_im(src_img)
    # vis_torch_im(src_mask)
    if mode == 'column':
        dim = 2
    elif mode == 'row':
        dim = 3

    dst_img = torch.zeros_like(src_img)
    mean_val = torch.sum(src_img * src_mask, dim=dim, keepdim=True) / (torch.sum(src_mask, dim=dim, keepdim=True) +1e-6)
    dst_img[0] = mean_val[0]

    dst_img = dst_img * dst_mask
    #
    # vis_torch_im(fit_bbox(src_mask))
    # vis_torch_im((src_mask))
    #
    if src_mask_close is None:
        src_mask_close = morpho_close_pano(src_mask, k=51)


    eff_mask = torch.sum(src_mask_close, dim=dim, keepdim=True) / (torch.sum(fit_bbox(src_mask), dim=dim, keepdim=True) + 1e-6)

    eff_mask =  (eff_mask + torch.zeros_like(src_mask))
    eff_mask = (eff_mask > min_valid_ratio) & (eff_mask < 2)



    eff_mask = eff_mask.float()
    dst_img = dst_img * eff_mask.float()
    return dst_img, eff_mask


def fill_nearest(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output:
        Return a filled array.
    """
    #import numpy as np
    #import scipy.ndimage as nd

    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]

def remove_small_gap(mask, gap_width=13):
    close_mask = morpho_close_pano(mask, gap_width)
    # close_mask = kornia.morphology.erosion(close_mask, torch.ones((gap_width, gap_width)))
    # vis_torch_im(close_mask)

    return close_mask

def fill_im(src_img, src_mask, dst_mask, thresh=0.2, min_valid_ratio=0.2, use_main_color=False):
    # src_img = src_img * src_mask
    # vis_torch_im(src_mask)
    dst_mask = dst_mask.float()
    src_mask_close = morpho_close_pano(src_mask, k=31)


    row_img, row_mask = _fill_im(src_img, src_mask, dst_mask, 'row', min_valid_ratio, src_mask_close=src_mask_close)
    col_img, col_mask = _fill_im(src_img, src_mask, dst_mask, 'column', min_valid_ratio, src_mask_close=src_mask_close)

    # vis_torch_im(row_img)
    # vis_torch_im(col_img)


    dst_img = torch.minimum(row_img, col_img)

    dst_img = (dst_img > thresh).float()
    # print(torch.unique(dst_img))

    # fill in the part outside the bbox
    fill_mask = 1 - fit_bbox(dst_img).float()
    # vis_torch_im(col_mask)
    # vis_torch_im(row_mask)

    dst_img_tmp = (torch.sum(dst_img, dim=2, keepdim=True) > 10).float()

    dst_img = ((dst_img_tmp * fill_mask + dst_img) > 0).float() * dst_mask
    # vis_torch_im(dst_img)
    fill_mask = 1 - fit_bbox(dst_img).float()
    # vis_torch_im(col_mask)
    # vis_torch_im(row_mask)

    dst_img_tmp = (torch.sum(dst_img, dim=3, keepdim=True) > 10).float()
    dst_img = ((dst_img_tmp * fill_mask + dst_img) > 0).float() * dst_mask

    dst_img = remove_small_gap(dst_img)

    if use_main_color:
        color = get_dominant_color(torch2np_im(src_img), torch2np_im(src_mask))

        dst_img = dst_img * torch.Tensor(color)[None, :, None, None]


    return dst_img


def render_sunlight_ratio(opt, sun_new, sunlight_i, sem_i, ambi_i, bright_wall_tex, bright_wall_mesh_dir, dark_wall_mesh_dir, bright_rt_wall_mesh_dir, dark_rt_wall_mesh_dir, normal, mode, sun_ratio_thresh=0.3 ):
    '''
    Render sunlight on floor or wall with color matching
    '''
    if mode == 'floor':
        # does not change the texture (if needed) at this time because this rendering is just for color matching
        Sun_img = render_plastic_sun(opt, 0.03, sun_new, bright_wall_tex, bright_wall_mesh_dir, dark_wall_mesh_dir,bright_rt_wall_mesh_dir, dark_rt_wall_mesh_dir,
                                   samples=opt.sun_samples, maxd=4, normal=normal)
    elif mode == 'wall':
        Sun_img = render_diffuse_sun(opt, sun_new, bright_wall_tex, bright_wall_mesh_dir, dark_wall_mesh_dir, bright_rt_wall_mesh_dir, dark_rt_wall_mesh_dir,
                                   samples=opt.sun_samples, maxd=4, normal=normal)

    Sun_img = Sun_img ** (1 / 2.2)
    Sun_img_Mask =  rgb2luma_torch(Sun_img)> 0.1

    sunlight_i = (sunlight_i * sem_i)
    sunlight_i_mask = rgb2luma_torch(sunlight_i) > sun_ratio_thresh

    Sun_img_matched = color_matching(Sun_img / ambi_i, Sun_img_Mask, sunlight_i, sunlight_i_mask)
    # vis_torch_im(sunlight_i_mask)
    # vis_torch_im(Sun_img_Mask.float())
    #
    # vis_torch_im(Sun_img)
    # vis_torch_im(Sun_img_matched)
    # vis_torch_im(ambi_i)
    color1 = get_dominant_color(torch2np_im(Sun_img_matched), torch2np_im(sunlight_i_mask))
    color2 = get_dominant_color(torch2np_im(Sun_img / (ambi_i+1e-6)), torch2np_im(sunlight_i_mask))


    # im1 = sunlight_i_floor_mask * torch.Tensor(color1)[None, :, None, None]
    # im2 = sunlight_i_floor_mask * torch.Tensor(color2)[None, :, None, None]

    ratio = rgb2luma(color1) / rgb2luma(color2)
    print(color1)
    print(color2)

    print(ratio)


    if mode == 'floor':
        # change the texture (if needed) since the ratio for color matching is estimated
        Sun_img = render_plastic_sun(opt, 0.03, sun_new, bright_wall_tex, bright_wall_mesh_dir, dark_wall_mesh_dir, bright_rt_wall_mesh_dir, dark_rt_wall_mesh_dir,
                                   samples=opt.sun_samples, floor_tex_dir=opt.floor_texture, maxd=4, normal=normal, ratio=ratio)
    elif mode == 'wall':
        Sun_img = render_diffuse_sun(opt, sun_new, bright_wall_tex * 5, bright_wall_mesh_dir, dark_wall_mesh_dir, bright_rt_wall_mesh_dir, dark_rt_wall_mesh_dir,
                                   samples=opt.sun_samples, maxd=4, normal=normal, ratio=ratio)

    Sun_img = Sun_img ** (1 / 2.2)

    return Sun_img


if __name__ == '__main__':
    render = torch.from_numpy(cv2.imread(f'out/render.exr', -1)[:, :, [2,1,0]]).permute(2,0,1)[None]
    vis_torch_im(render * 400 * 3.14 / 10)

    # im = read_im('test.png') * 2
    # vis_torch_im(im)
    # mask_floor = read_im('mask_floor.png')
    # mask_wall = read_im('mask_wall.png')
    # # mask_floor = morpho_erode_pano(mask_floor, k=31)
    # # mask_wall = morpho_erode_pano(mask_wall, k=11)
    #
    # tex_fit_mask = fit_bbox(im)
    #
    # filled_im1 = fill_im(im, mask_wall, tex_fit_mask, 0.2)
    # # filled_im2 = fill_im(im, mask_floor, tex_fit_mask,  0.4)
    # color = get_dominant_color(torch2np_im(im), torch2np_im(mask_wall))
    #
    # img = filled_im1 * torch.Tensor(color)[None, :, None, None]
    # vis_torch_im(img)