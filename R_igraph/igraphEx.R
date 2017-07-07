##### igraph Ex #####

library(igraph)
library(rgl)
library(ape)
setwd("/Users/yumeng.zou/Downloads")
igraphdemo()

## Chinese characters
nodes<-read.csv("10mName.csv",encoding="UTF-8",stringsAsFactors=F)
# encoding="UTF-8" -- for Chinese characters
# stringsAsFactors=F -- for using characters as labels for nodes
####sessionInfo()
Sys.setlocale(category = "LC_CTYPE", locale = "cht") 
# "chs" for simplified Chinese
head(nodes)

## Create a graph
edges<-read.csv("10mEdge.csv")
el<-cbind(edges$Source,edges$Target)
el<-el+1 # R index starts from 1, not 0
g<-graph_from_edgelist(el,directed=F)
summary(g)
plot(g)

## Vertices and Edges
V(g)
E(g)

## Change Attributes of the graph
g$layout<-layout_in_circle
V(g)$label<-nodes$Label
V(g)$label.cex<-1
V(g)$size<-3
plot(g)

## Structural properties of vertices
degree(g)
plot(degree_distribution(g),type='l')
centr_degree(g)
centr_betw(g,directed=F)
centr_clo(g,mode='all')
centr_eigen(g)

## Without ???
#### layout
g2<-delete_vertices(g,190)
g2$layout<-layout_with_fr
plot(g2)
g2$layout<-layout_as_tree
plot(g2)
g2$layout<-layout_as_star
plot(g2)
g2$layout<-layout_on_grid
plot(g2)
g2$layout<-layout_with_kk
plot(g2)
tkplot(g2)

## Structural Properties
transitivity(g2)
transitivity(g2,type='local')
components(g2)
centr_degree(g2)
centr_betw(g2,directed=F)
centr_clo(g2,mode='all')
centr_eigen(g2)
cliques(g2,min=4)
eccentricity(g2)



