

def plot_2d_embeddings(embeddings_2d, model_name, reducer_name):

    title = f"{model_name} Embeddings per Class by {reducer_name}"
    save_filepath = f'{root}figures/{model_name}_{reducer_name}_train.png'

    color_map = {0: "lightpink", 1: "lightblue"}
    rf_label_map = {0: "GEN", 1: "REAL"}
    centroid_color_map = {0: "red", 1: "blue"}
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i in range(num_class):
        ax = axes[i]
        area_dict = {}
        
        for rf in [1, 0]: # "0:FAKE", "1:REAL"
            subset = (labels == i) & (reals == rf)
            ax.scatter(
                embeddings_2d[subset, 0],
                embeddings_2d[subset, 1],
                label=rf_label_map[rf],
                color=color_map[rf],
                alpha=0.3,
                s=15
            )
    
            # centroid
            if subset.sum() > 0:
                centroid = embeddings_2d[subset].mean(axis=0)
                ax.scatter(
                    centroid[0], centroid[1],
                    color=centroid_color_map[rf],
                    marker='X',
                    s=100,
                    linewidth=0.8,
                    label=f"{rf_label_map[rf]} Centroid"
                )
    
                # radius: mean point of the mean point clouds
                dists = ((embeddings_2d[subset] - centroid)**2).sum(axis=1)**0.5
                radius = dists.mean()
                area_dict[rf] = np.pi * (radius ** 2) # area
    
                # circle
                circle = patches.Circle(
                    (centroid[0], centroid[1]),
                    radius,
                    color=centroid_color_map[rf],
                    alpha=0.2,
                    linestyle='--',
                    linewidth=1,
                    fill=True
                )
                ax.add_patch(circle)
            
        ax.set_title(f"{id_to_class[i]}")
        #if 0 in area_dict and 1 in area_dict:
        #    area_diff = area_dict[1] / area_dict[0] # fake / real
        #    ax.set_title(f"{id_to_class[i]}\nArea = {area_diff:.2f}", fontsize=14)
        #else:
        #    ax.set_title(f"{id_to_class[i]}", fontsize=14)
            
        ax.grid(True)
        ax.legend(fontsize=10)
    
    #plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_filepath)
    plt.show()