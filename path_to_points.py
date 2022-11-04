from svg.path import parse_path
import numpy as np
from xml.dom import minidom
from math import ceil



def get_point_at(path, distance, scale, offset):
    pos = path.point(distance)
    pos += offset
    pos *= scale
    return pos.real, pos.imag


def points_from_path(path, density, scale, offset):
    step = int(path.length() * density)
    last_step = step - 1

    if last_step == 0:
        yield get_point_at(path, 0, scale, offset)
        return

    for distance in range(step):
        yield get_point_at(
            path, distance / last_step, scale, offset
        )


def sorting(segments):
    n_segments = len(segments)
    end_points = []
    
    for seg in segments:
        end_points += [seg[0], seg[-1]]

    end_points = np.array(end_points)
    distance = np.linalg.norm(
        end_points - np.expand_dims(end_points, axis=1),
        axis=-1
    )
    inf_x_ids = np.repeat(np.arange(distance.shape[0]), 2, axis=0), 
    inf_y_ids = np.repeat(np.arange(distance.shape[0]).reshape(-1, 2), 2, axis=0).flatten()
    distance[inf_x_ids, inf_y_ids] = np.inf
    sorted_dist = np.argsort(distance, axis=0)

    sorted_ids = []
    directions = []
    min_distance = np.min(distance, axis=0)
    sorted_ids.append(np.argmax(min_distance) // 2)
    directions.append(np.argmax(min_distance) % 2) #0: forward, 1: invert

    for i in range(n_segments-1):
        anchor_id = sorted_ids[-1] * 2 + (1 - directions[-1])
        for j in sorted_dist[:, anchor_id]:
            j_segment = j // 2
            if j_segment not in sorted_ids:
                sorted_ids.append(j_segment)
                directions.append(j % 2)
                break

    new_segments = [
        segments[i] if d == 0 else segments[i][::-1]
        for i, d in zip(sorted_ids, directions)
    ]

    return new_segments


def segments_from_doc(doc, density=5, scale=1, offset=0):
    offset = offset[0] + offset[1] * 1j
    segments = []
    for element in doc.getElementsByTagName("path"):
        points = []
        for path in parse_path(element.getAttribute("d")):
            points.extend(
                points_from_path(path, density, scale, offset)
            )
        segments.append(points)

    segments = sorting(segments)
    
    return segments


def gen_points_between_2_points(p1, p2, points_per_pixel):
    delta = np.linalg.norm((p1[0]-p2[0], p1[1]-p2[1]))
    n_extra_point = ceil(delta*points_per_pixel) - 1
    xs_extra = np.linspace(p1[0], p2[0], n_extra_point+2).tolist()
    ys_extra = np.linspace(p1[1], p2[1], n_extra_point+2).tolist()

    return xs_extra, ys_extra

def segments_to_points(segments, points_per_pixel):
    xs = []
    ys = []
    t_valid = []
    pre_point = segments[-1][-1]
    for seg in segments:
        xs_extra, ys_extra = gen_points_between_2_points(pre_point, seg[0], points_per_pixel)
        pre_point = seg[-1]
        xs += xs_extra
        ys += ys_extra
        t_valid += [0] * len(xs_extra)
        t_valid += [1] * len(seg)
        for point in seg:
            xs.append(float(point[0]))
            ys.append(float(point[1]))

    points = np.empty((len(xs), ), dtype=np.complex128)
    points.real = xs
    points.imag = ys
    

    return points, t_valid


def path_to_points(path_name, points_per_pixel=0.2):

    with open(path_name, 'rb') as f:
        string = f.read()
   
    doc = minidom.parseString(string)
    segments = segments_from_doc(doc, points_per_pixel, 1, (0, 0))
    doc.unlink()
    points, t_valid = segments_to_points(segments, points_per_pixel)

    return points, t_valid

