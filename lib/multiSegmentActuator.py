import xml.etree.cElementTree as ET
import xml.dom.minidom
import numpy as np


def generateXML(config, nSprings=3):
    
    dx = config['fillRatio'] * config['length'] / (2 * config['numSegments'] * (config['numDisks']-1))
    
    m = config['mass'] / (config['numSegments'] * (config['numDisks']-1) + 1)
    
    root = ET.Element('mujoco', model='{}'.format(config['file']))
    opt = ET.SubElement(root, 'option', timestep='{}'.format(config['timeStep']), integrator='implicitfast', gravity='0 0 0')
    ET.SubElement(opt, 'flag', energy='enable')
    vis = ET.SubElement(root, 'visual')
    ET.SubElement(vis, 'global', offwidth="4096", offheight='4096')
    body = ET.SubElement(root, 'worldbody')
    ET.SubElement(body, 'camera', name = 'mainCamera', pos='{} {} {}'.format(*config['camPosition']), euler='{} {} {}'.format(*config['camOrientation']))

    for kk in range(config['numSegments']):
        for jj in range(config['numDisks']):
            if jj == config['numDisks']-1 and kk != config['numSegments'] - 1:
                continue
            body = ET.SubElement(body, 'body', name = 'A{:02d}B{:02d}'.format(kk, jj), pos = '{} 0 0'.format(2 * dx / config['fillRatio'] if  (kk > 0 or jj > 0) else 0))
            ET.SubElement(body, 'geom', name = 'A{:02d}G{:02d}'.format(kk, jj), type='cylinder', size='{} {}'.format(config['radius'], dx), rgba='{} {} {} 1'.format(*(np.ones(3) * 0.8 - np.ones(3) * 0.4 * (jj==0))), euler='0 90 0')
            ET.SubElement(body, 'inertial', pos='0 0 0', mass='{}'.format(m), diaginertia='{} {} {}'.format(m * (3 * config['radius']**2 + 4 * dx**2) / 12, m * (3 * config['radius']**2 + 4 * dx**2) / 12, m * (config['radius']**2) / 2))
            ET.SubElement(body, 'site', name = 'A{:02d}B{:02d}S{:02d}C'.format(kk, jj, 0), size = '0.0001', pos = '0 0 0')
            for ii in range(nSprings):
                ET.SubElement(body, 'site', name = 'A{:02d}B{:02d}S{:02d}C'.format(kk, jj, ii+1), size = '0.0001', pos = '0 {} {}'.format(config['radius'] * np.cos(2 * np.pi * ii / nSprings).round(12), config['radius'] * np.sin(2 * np.pi * ii / nSprings).round(12)))      
            if  (kk > 0 or jj > 0):
                ET.SubElement(body, 'joint', name = 'A{:02d}R{:02d}y'.format(kk, jj), pos = '0 0 0', type='hinge', axis='0 1 0')
                ET.SubElement(body, 'joint', name = 'A{:02d}R{:02d}z'.format(kk, jj), pos = '0 0 0', type='hinge', axis='0 0 1')
                ET.SubElement(body, 'joint', name = 'A{:02d}P{:02d}x'.format(kk, jj), pos = '0 0 0', type='slide', axis='1 0 0')
    '''
    part = ET.SubElement(part, 'body', euler='0 -90 0', pos='-0.025 0 0')
    rep = ET.SubElement(part, 'replicate', count='10', euler='0 0 36')
    frame = ET.SubElement(rep, 'frame', pos='-0.0001 0 .03', euler='0 15 0')
    rep2 = ET.SubElement(frame, 'replicate', count='3', euler='0 10 0')
    ET.SubElement(rep2, 'geom', type='box', size='.0065 .003 .001', pos='0 0 -0.075', rgba='0.8 0.1 0.1 1')
    '''
    tend = ET.SubElement(root, 'tendon')
    for kk in range(config['numSegments']):
        r_ = 2*kk/(config['numSegments']-1) if config['numSegments'] > 1 else 0 
        for ii in range(1, nSprings + 1):
            for jj in range(config['numDisks'] - 1):
                isLast = (jj==config['numDisks']-2) * (kk != config['numSegments']-1)
                sc = ET.SubElement(tend, 'spatial', name='A{:02d}B{:02d}T{:02d}'.format(kk, jj, ii), stiffness='{}'.format(config['innerStiffness'] * (config['numDisks']-1)), damping='{}'.format(config['innerDamping'] * (config['numDisks']-1)), width='0.001', rgba='{} {} {} 1'.format(0.6*min(2 - r_, 1)**2, 0.6*min(2 - r_, r_)**2, 0.6*min(r_, 1)**2))
                ET.SubElement(sc, 'site', site='A{:02d}B{:02d}S{:02d}C'.format(kk, jj, ii))
                ET.SubElement(sc, 'site', site='A{:02d}B{:02d}S{:02d}C'.format(kk + isLast, (jj + 1) % (config['numDisks'] - isLast), ii))

    for kk in range(config['numSegments']):
        r_ = 2*kk/(config['numSegments']-1) if config['numSegments'] > 1 else 0
        for ii in range(1, nSprings + 1):
            for jj in range(config['numDisks'] - 1):
                isLast = (jj==config['numDisks']-2) * (kk != config['numSegments']-1)
                
                sc = ET.SubElement(tend, 'spatial', name='A{:02d}B{:02d}T{:02d}_'.format(kk, jj, ii), stiffness='{}'.format(config['outerStiffness'] * (config['numDisks']-1)), damping='{}'.format(config['outerDamping'] * (config['numDisks']-1)), width='0.001', rgba='{} {} {} 1'.format(0.6*min(2 - r_, 1)**2, 0.6*min(2 - r_, r_)**2, 0.6*min(r_, 1)**2))
                ET.SubElement(sc, 'site', site='A{:02d}B{:02d}S{:02d}C'.format(kk, jj, ii))
                ET.SubElement(sc, 'site', site='A{:02d}B{:02d}S{:02d}C'.format(kk + isLast, (jj + 1) % (config['numDisks'] - isLast), ii))

    for kk in range(config['numSegments']):
        r_ = 2*kk/(config['numSegments']-1) if config['numSegments'] > 1 else 0
        sc = ET.SubElement(tend, 'spatial', name='T{}'.format(kk), stiffness='0', damping='0', width='0.01', rgba='{} {} {} 1'.format(0.6*min(2 - r_, 1)**2, 0.6*min(2 - r_, r_)**2, 0.6*min(r_, 1)**2))
        for jj in range(config['numDisks']):
            isLast = (jj==config['numDisks']-1) * (kk != config['numSegments']-1)
            ET.SubElement(sc, 'site', site='A{:02d}B{:02d}S{:02d}C'.format(kk + isLast, jj * (1 - isLast), 0))

    act = ET.SubElement(root, 'actuator')

    for kk in range(config['numSegments']):
        ET.SubElement(act, 'general', tendon='T{}'.format(kk), ctrllimited='true', ctrlrange='0 1', gear='{}'.format(config['gear']))
    
    dom = xml.dom.minidom.parseString(ET.tostring(root))
    xml_string = dom.toprettyxml()

    with open('{}'.format(config['file']), 'w') as xfile:
        xfile.write(xml_string)
        xfile.close()