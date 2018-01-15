import tables
f = tables.open_file("sif.h5", "r+")
f.remove_node(f.root.sif.word_weight)
f.close()
