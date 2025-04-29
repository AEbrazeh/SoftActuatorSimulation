import xml.etree.cElementTree as ET
import xml.dom.minidom
import numpy as np


def createSegment(name, nSegments, L, r, N, innerStiffness, innerDamping, nInnerSprings, gear, mass = 0.1, dt = 0.01, percentage = 0.5):
    DF = 2 / percentage - 2
    dx = L/(N * nSegments * (DF + 2) - DF)
    root = ET.Element('mujoco', model='{}'.format(name))
    ET.SubElement(root, 'option', timestep='{}'.format(dt), integrator='implicitfast', gravity='0 0 0')
    vis = ET.SubElement(root, 'visual')
    ET.SubElement(vis, 'global', offwidth="4096", offheight='4096')
    wb = ET.SubElement(root, 'worldbody')
    ET.SubElement(wb, 'camera', pos='{} 0 1'.format(L), name = 'mainCamera', euler='0 0 0')
    
    part = ET.SubElement(wb, 'body', name = 'B{:02d}'.format(0), pos = '{} 0 0'.format(0))
    ET.SubElement(part, 'geom', name = 'A{:02d}G{:02d}'.format(0, 0), type='cylinder', size='{} {}'.format(r, dx), rgba='0.8 0.1 0.1 1', euler='0 90 0')
    ET.SubElement(part, 'inertial', pos='0 0 0', mass='{}'.format(mass / (2*N)), diaginertia='{} {} {}'.format(mass / (2*N) * (3 * r**2 + dx**2) / 12, mass / (2*N) * (3 * r**2 + dx**2) / 12, mass / (2*N) * (r**2) / 2))
    ET.SubElement(part, 'site', name = 'A{:02d}B{:02d}S{:02d}C'.format(0, 0, 0), size = '0.0001', pos = '{} 0 0'.format(0))
    for ii in range(nInnerSprings):
        ET.SubElement(part, 'site', name = 'A{:02d}B{:02d}S{:02d}C'.format(0, 0, ii+1), size = '0.0001', pos = '{} {} {}'.format(0, r * np.cos(2 * np.pi * ii / nInnerSprings).round(12), r * np.sin(2 * np.pi * ii / nInnerSprings).round(12)))
    for kk in range(nSegments):
        for jj in range(N):
            if kk == 0 and jj == 0: continue
            part = ET.SubElement(part, 'body', name = 'A{:02d}B{:02d}'.format(kk, jj), pos = '{} 0 0'.format((DF + 2) * dx))
            ET.SubElement(part, 'geom', name = 'A{:02d}G{:02d}'.format(kk, jj), type='cylinder', size='{} {}'.format(r, dx), rgba='{} {} {} 1'.format(0.8, 0.1 if (jj % (N+1)) == 0 else 0.8, 0.1 if (jj % (N+1)) == 0 else 0.8), euler='0 90 0')
            ET.SubElement(part, 'inertial', pos='0 0 0', mass='{}'.format(mass / (2*N)), diaginertia='{} {} {}'.format(mass / (2*N) * (3 * r**2 + dx**2) / 12, mass / (2*N) * (3 * r**2 + dx**2) / 12, mass / (2*N) * (r**2) / 2))
            ET.SubElement(part, 'site', name = 'A{:02d}B{:02d}S{:02d}C'.format(kk, jj, 0), size = '0.0001', pos = '{} 0 0'.format(0))
            for ii in range(nInnerSprings):
                ET.SubElement(part, 'site', name = 'A{:02d}B{:02d}S{:02d}C'.format(kk, jj, ii+1), size = '0.0001', pos = '{} {} {}'.format(0, r * np.cos(2 * np.pi * ii / nInnerSprings).round(12), r * np.sin(2 * np.pi * ii / nInnerSprings).round(12)))      
            if jj != 0:
                ET.SubElement(part, 'joint', name = 'A{:02d}P{:02d}x'.format(kk, jj), pos = '0 0 0', type='slide', axis='1 0 0')
                ET.SubElement(part, 'joint', name = 'A{:02d}R{:02d}y'.format(kk, jj), pos = '0 0 0', type='hinge', axis='0 1 0')
                ET.SubElement(part, 'joint', name = 'A{:02d}R{:02d}z'.format(kk, jj), pos = '0 0 0', type='hinge', axis='0 0 1')
    
    '''
    part = ET.SubElement(part, 'body', euler='0 -90 0', pos='-0.025 0 0')
    rep = ET.SubElement(part, 'replicate', count='10', euler='0 0 36')
    frame = ET.SubElement(rep, 'frame', pos='-0.0001 0 .03', euler='0 15 0')
    rep2 = ET.SubElement(frame, 'replicate', count='3', euler='0 10 0')
    ET.SubElement(rep2, 'geom', type='box', size='.0065 .003 .001', pos='0 0 -0.075', rgba='0.8 0.1 0.1 1')
    '''
    tend = ET.SubElement(root, 'tendon')
    for kk in range(nSegments):
        for ii in range(1, nInnerSprings + 1):
            for jj in range(N - 1):
                sc = ET.SubElement(tend, 'spatial', name='A{:02d}B{:02d}T{:02d}'.format(kk, jj, ii), stiffness='{}'.format(innerStiffness * (N - 1) * nSegments / nInnerSprings), damping='{}'.format(innerDamping * (N - 1) * nSegments / nInnerSprings), width='0.001', rgba='0.8 0.0 0.0 1')
                ET.SubElement(sc, 'site', site='A{:02d}B{:02d}S{:02d}C'.format(kk, jj, ii))
                ET.SubElement(sc, 'site', site='A{:02d}B{:02d}S{:02d}C'.format(kk, jj+1, ii))

    
    sc = ET.SubElement(tend, 'spatial', name='T0', stiffness='0', damping='0', width='0.01', rgba='0.8 0 0 1')
    for kk in range(nSegments):
        for jj in range(N):
            ET.SubElement(sc, 'site', site='A{:02d}B{:02d}S{:02d}C'.format(kk, jj, 0))
            
    act = ET.SubElement(root, 'actuator')

    for kk in range(nSegments):
        for ii in range(1, nInnerSprings + 1):
            for jj in range(N - 1):
                ET.SubElement(act, 'general', tendon='A{:02d}B{:02d}T{:02d}'.format(kk, jj, ii), ctrllimited='false')
    ET.SubElement(act, 'general', tendon='T0', ctrllimited='true', ctrlrange='0 1', gear='{}'.format(gear))
    
    dom = xml.dom.minidom.parseString(ET.tostring(root))
    xml_string = dom.toprettyxml()

    with open('{}.xml'.format(name), 'w') as xfile:
        xfile.write(xml_string)
        xfile.close()