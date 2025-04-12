from grad_cam import get_gradcam  


test_img = r"C:\Users\Manoj\dsa38\project\webapp\static\uploads\Covid (224).png"  
heatmap_path = get_gradcam(test_img)
print("Saved Grad-CAM at:", heatmap_path)
