import dimsim

# 计算词语间的发音相似度
candidates = dimsim.get_distance("牛奶",'溜来')
print(candidates)