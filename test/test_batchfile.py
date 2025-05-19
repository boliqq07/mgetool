from pathlib import Path


from mgetool.imports.batchfilematch import BatchFileMatch,BatchPathMatch


bf = BatchPathMatch(r"D:\PycharmProjects\mgetool\test\bf")

# bf.filter_dir_name(exclude="sf2")
# bf.filter_file_name(include="asf.bmp")
# bf.filter_file_name_parent_folder(exclude="asf.bmp")
# bf.merge()
print(bf.file_dir)
# print(bf.file_list)
