matrix_file <- "E:\\Work\\MBS\\EEG\\Modelling\\Intelligence\\matrix_glass_brain.csv"
matrix <- t(as.matrix(read.csv(matrix_file)))
atlas_file <- "E:\\Work\\MBS\\EEG\\Modelling\\Intelligence\\Destrieux atlas.csv"
atlas = read.csv(atlas_file)
brainconn(atlas=atlas, conmat=matrix, view="left", labels = F, label.size = 5, edge.alpha = 0.5, show.legend = F)
ggsave("E:\\Work\\MBS\\EEG\\Modelling\\Intelligence\\Results\\Imgs for article\\Lesion aggr\\For glass brain\\picture.png", bg="white")