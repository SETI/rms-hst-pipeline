import os.path

import FileArchive
import FileArchives

def quadrupleToNewFilepath(inst, prop, vis, f):
    bundlePart = 'hst_%d' % prop

    noExt = os.path.splitext(f)[0]
    parts = noExt.split('_')
    suffix = '_'.join(parts[1:])
    collectionPart = 'data_%s_%s' % (inst, suffix)

    visitPart = 'visit_%s' % vis
    return '/'.join([bundlePart, collectionPart, visitPart, f])

def shuffle(arch):
    for (inst, prop, vis, f) in arch.walkFiles():
        src = '/'.join([arch.visitFilepath(inst, prop, vis), f])
        dst = '/'.join([arch.root, quadrupleToNewFilepath(inst, prop, vis, f)])
        print 'mv %s %s' % (src, dst)

shuffle(FileArchives.getAnyArchive())
