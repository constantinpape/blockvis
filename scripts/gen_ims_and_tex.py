import vigra
import numpy as np
import os

def nodes_and_edges_for_subgraph(raw, seg, x_min, x_max, y_min, y_max, save_generated_folder):

    seg = vigra.analysis.labelImage(seg) - 1

    xy_mask = np.zeros(seg.shape, np.bool)
    xy_mask[x_min:x_max, y_min:y_max] = True

    sX = seg.shape[0]
    sY = seg.shape[1]

    min_plot_x = float(x_min) / sX
    max_plot_x = float(x_max) / sX
    min_plot_y = 1. - float(y_min) / sY
    max_plot_y = 1. - float(y_max) / sY
    # draw subvol boundary
    subvolstr = "\draw[very thick] (%f,%f) rectangle (%f,%f);"%(min_plot_x, min_plot_y, max_plot_x, max_plot_y)

    bb_tag = "bb%ito%i-%ito%i" % (x_min, x_max, y_min, y_max)

    if not os.path.exists(save_generated_folder):
        os.mkdir(save_generated_folder)

    with open( os.path.join(save_generated_folder,"subvol_%s.tex" % bb_tag), 'w') as f:
        f.write(subvolstr)

    # get the region centers
    nodeIds = np.unique(seg)
    nodeCenters = vigra.filters.eccentricityCenters(seg)

    nodeStr = ""
    edgeStr = ""

    # generate tikz nodes

    node_inout = []
    for nodeId in nodeIds:
        coord = np.round(nodeCenters[nodeId],2)
        x = float(coord[0]) / float(sX)
        y = float(coord[1]) / float(sY)
        y = 1.0 - y

        # look if node is in subvolume
        node_coords = np.where(seg == nodeId)
        x_coord = np.sort(node_coords[0])
        y_coord = np.sort(node_coords[1])

        inside = False

        node_mask = np.zeros(seg.shape, np.bool)
        node_mask[x_coord[0]:x_coord[-1], y_coord[0]:y_coord[-1]] = True

        ovlp = np.logical_and(node_mask, xy_mask)

        if np.sum(ovlp) > 0:
            inside = True

        if inside:
            nodeStr += "\\node[in_node] at (%f, %f) (n%d){};\n"%(x,y, nodeId)
        else:
            nodeStr += "\\node[out_node] at (%f, %f) (n%d){};\n"%(x,y, nodeId)
        node_inout.append(inside)

    with open(os.path.join(save_generated_folder,"nodes_sub_%s.tex" % bb_tag), 'w') as f:
        f.write(nodeStr)

    # generate tikz edges
    rag = vigra.graphs.regionAdjacencyGraph(vigra.graphs.gridGraph(seg.shape[0:2]), seg)
    uv = rag.uvIds()
    for edgeId in xrange(uv.shape[0]):
        u = uv[edgeId,0]
        v = uv[edgeId,1]
        if node_inout[u] and node_inout[v]:
            edgeStr += "\draw[edge_in] (n%d) -- (n%d);\n"%(u,v)
        elif node_inout[u] and not node_inout[v]:
            edgeStr += "\draw[edge_out] (n%d) -- (n%d);\n"%(u,v)
        elif not node_inout[u] and node_inout[v]:
            edgeStr += "\draw[edge_out] (n%d) -- (n%d);\n"%(u,v)
        elif not node_inout[u] and not node_inout[v]:
            edgeStr += "\draw[edge_remaining] (n%d) -- (n%d);\n"%(u,v)

    with open(os.path.join(save_generated_folder,"edges_sub_%s.tex" % bb_tag), 'w') as f:
        f.write(edgeStr)


def nodes_and_edges_for_subgraph_cut(raw, seg, node_result, x_min, x_max, y_min, y_max,save_generated_folder):

    seg, _, mapping = vigra.analysis.relabelConsecutive(seg, start_label = 0)
    reversed_mapping = {mapping[k] : k for k in mapping}

    xy_mask = np.zeros(seg.shape, np.bool)
    xy_mask[x_min:x_max, y_min:y_max] = True

    sX = seg.shape[0]
    sY = seg.shape[1]

    min_plot_x = float(x_min) / sX
    max_plot_x = float(x_max) / sX
    min_plot_y = 1. - float(y_min) / sY
    max_plot_y = 1. - float(y_max) / sY
    # draw subvol boundary
    subvolstr = "\draw[very thick] (%f,%f) rectangle (%f,%f);"%(min_plot_x, min_plot_y, max_plot_x, max_plot_y)

    bb_tag = "bb%ito%i-%ito%i" % (x_min, x_max, y_min, y_max)

    if not os.path.exists(save_generated_folder):
        os.mkdir(save_generated_folder)

    with open( os.path.join(save_generated_folder,"subvol_%s.tex" % bb_tag), 'w') as f:
        f.write(subvolstr)

    # get the region centers
    nodeIds = np.unique(seg)
    nodeCenters = vigra.filters.eccentricityCenters(seg)

    nodeStr = ""
    edgeStr = ""

    # generate tikz nodes

    node_inout = []
    for nodeId in nodeIds:
        coord = np.round(nodeCenters[nodeId],2)
        x = float(coord[0]) / float(sX)
        y = float(coord[1]) / float(sY)
        y = 1.0 - y

        # look if node is in subvolume
        node_coords = np.where(seg == nodeId)
        x_coord = np.sort(node_coords[0])
        y_coord = np.sort(node_coords[1])

        inside = False

        node_mask = np.zeros(seg.shape, np.bool)
        node_mask[x_coord[0]:x_coord[-1], y_coord[0]:y_coord[-1]] = True

        ovlp = np.logical_and(node_mask, xy_mask)

        if np.sum(ovlp) > 0:
            inside = True

        if inside:
            nodeStr += "\\node[in_node] at (%f, %f) (n%d){};\n"%(x,y, nodeId)
        else:
            nodeStr += "\\node[out_node] at (%f, %f) (n%d){};\n"%(x,y, nodeId)
        node_inout.append(inside)

    with open(os.path.join(save_generated_folder,"nodes_sub_%s.tex" % bb_tag), 'w') as f:
        f.write(nodeStr)

    # generate tikz edges
    rag = vigra.graphs.regionAdjacencyGraph(vigra.graphs.gridGraph(seg.shape[0:2]), seg)
    uv = rag.uvIds()
    for edgeId in xrange(uv.shape[0]):
        u = uv[edgeId,0]
        v = uv[edgeId,1]
        is_cut = node_result[reversed_mapping[u]] == node_result[reversed_mapping[v]]
        if node_inout[u] and node_inout[v]:
            if is_cut:
                edgeStr += "\draw[edge_in_cut] (n%d) -- (n%d);\n"%(u,v)
            else:
                edgeStr += "\draw[edge_in] (n%d) -- (n%d);\n"%(u,v)
        elif node_inout[u] and not node_inout[v]:
            edgeStr += "\draw[edge_out] (n%d) -- (n%d);\n"%(u,v)
        elif not node_inout[u] and node_inout[v]:
            edgeStr += "\draw[edge_out] (n%d) -- (n%d);\n"%(u,v)
        elif not node_inout[u] and not node_inout[v]:
            edgeStr += "\draw[edge_remaining] (n%d) -- (n%d);\n"%(u,v)

    with open(os.path.join(save_generated_folder,"edges_sub_%s.tex" % bb_tag), 'w') as f:
        f.write(edgeStr)


def merge_seg_in_bb(raw, seg, node_result, x_min, x_max, y_min, y_max):
    import nifty
    seg, _, mapping = vigra.analysis.relabelConsecutive(seg, start_label = 0)
    reversed_mapping = {mapping[k] : k for k in mapping}
    xy_mask = np.zeros(seg.shape, np.bool)
    xy_mask[x_min:x_max, y_min:y_max] = True
    # get the region centers
    nodeIds = np.unique(seg)
    node_inout = []
    for nodeId in nodeIds:
        # look if node is in subvolume
        node_coords = np.where(seg == nodeId)
        x_coord = np.sort(node_coords[0])
        y_coord = np.sort(node_coords[1])
        inside = False
        node_mask = np.zeros(seg.shape, np.bool)
        node_mask[x_coord[0]:x_coord[-1], y_coord[0]:y_coord[-1]] = True
        ovlp = np.logical_and(node_mask, xy_mask)
        if np.sum(ovlp) > 0:
            inside = True
        node_inout.append(inside)
    # generate tikz edges
    rag = vigra.graphs.regionAdjacencyGraph(vigra.graphs.gridGraph(seg.shape[0:2]), seg)
    uv = rag.uvIds()
    ufd = nifty.ufd.ufd(rag.nodeNum)
    for edgeId in xrange(uv.shape[0]):
        u = uv[edgeId,0]
        v = uv[edgeId,1]
        is_cut = node_result[reversed_mapping[u]] == node_result[reversed_mapping[v]]
        if node_inout[u] and node_inout[v]:
            if is_cut:
                ufd.merge(u,v)
    nodeLabels = ufd.elementLabeling()
    mergedSeg = rag.projectLabelsToBaseGraph(nodeLabels.astype('uint32'))
    vigra.writeHDF5(mergedSeg, '../data/example_data.h5', 'subseg2')


def merged_graph(raw, seg, node_result):
    import nifty
    seg, _, mapping = vigra.analysis.relabelConsecutive(seg, start_label = 0)
    reversed_mapping = {mapping[k] : k for k in mapping}
    y_min = 0
    y_max = seg.shape[1]
    rag = vigra.graphs.regionAdjacencyGraph(vigra.graphs.gridGraph(seg.shape[0:2]), seg)
    uv = rag.uvIds()
    edge_states = np.zeros(rag.edgeNum, 'uint8')
    for x_min, x_max in ((0,384),(384,786)):
        xy_mask = np.zeros(seg.shape, np.bool)
        xy_mask[x_min:x_max, y_min:y_max] = True
        # get the region centers
        nodeIds = np.unique(seg)
        node_inout = []
        for nodeId in nodeIds:
            # look if node is in subvolume
            node_coords = np.where(seg == nodeId)
            x_coord = np.sort(node_coords[0])
            y_coord = np.sort(node_coords[1])
            inside = False
            node_mask = np.zeros(seg.shape, np.bool)
            node_mask[x_coord[0]:x_coord[-1], y_coord[0]:y_coord[-1]] = True
            ovlp = np.logical_and(node_mask, xy_mask)
            if np.sum(ovlp) > 0:
                inside = True
            node_inout.append(inside)
        # generate tikz edges
        for edgeId in xrange(uv.shape[0]):
            u = uv[edgeId,0]
            v = uv[edgeId,1]
            is_cut = node_result[reversed_mapping[u]] != node_result[reversed_mapping[v]]
            if node_inout[u] and node_inout[v]:
                if is_cut:
                    edge_states[edgeId] += 1
            elif node_inout[u] and not node_inout[v]:
                edge_states[edgeId] += 1
            elif not node_inout[u] and node_inout[v]:
                edge_states[edgeId] += 1

    ufd = nifty.ufd.ufd(rag.nodeNum)
    for edgeId in xrange(edge_states.shape[0]):
        if edge_states[edgeId] == 0:
            ufd.merge(uv[edgeId,0], uv[edgeId,1])
    nodeLabels = ufd.elementLabeling()
    mergedSeg = rag.projectLabelsToBaseGraph(nodeLabels.astype('uint32'))
    vigra.writeHDF5(mergedSeg, '../data/example_data.h5', 'mergedseg')


if __name__ == '__main__':
    raw = vigra.readHDF5('../data/example_data.h5', 'raw')
    seg = vigra.readHDF5('../data/example_data.h5', 'mergedseg')
    node_res = vigra.readHDF5('../data/sample_B_test_mc_defectsV2_nodes.h5', 'data')
    x_min = 0
    x_max = raw.shape[0]
    y_min = 0
    y_max = raw.shape[1]

    #merged_graph(raw, seg, node_res)
    nodes_and_edges_for_subgraph(raw, seg, x_min, x_max, y_min, y_max, '../data/generated_tex_4_spraw')
