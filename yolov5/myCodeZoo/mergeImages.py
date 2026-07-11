from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm

def horizontal_concatenate_images(folder1, folder2, folder3, folder4, output_folder):
    # 获取两个文件夹中的文件名
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)
    files3 = os.listdir(folder3)
    files4 = os.listdir(folder4)

    # 获取两个文件夹中共有的文件名
    common_files = set(files1).intersection(files2).intersection(files3).intersection(files4)

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in tqdm(common_files):
        # 拼接图片的完整路径
        path1 = os.path.join(folder1, file)
        path2 = os.path.join(folder2, file)
        path3 = os.path.join(folder3, file)
        path4 = os.path.join(folder4, file)

        # 打开两张图片
        image1 = Image.open(path1)
        image2 = Image.open(path2)
        image3 = Image.open(path3)
        image4 = Image.open(path4)

        # # 在拼接后的图片上写上文件夹的名字
        # draw = ImageDraw.Draw(image1)
        # draw2 = ImageDraw.Draw(image2)
        # font = ImageFont.truetype("arial.ttf", 32)
        # folder1_name = os.path.basename(folder1)
        # folder2_name = os.path.basename(folder2)
        # draw.text((10, 10), folder1_name, fill='red', font=font)
        # draw2.text((10, 10), folder2_name, fill='red', font=font)

        # 横向拼接图片
        result_image = Image.new('RGB', (image1.width + image2.width + image3.width + image4.width, image1.height))
        result_image.paste(image1, (0, 0))
        result_image.paste(image2, (image1.width, 0))
        result_image.paste(image3, (image1.width + image2.width, 0))
        result_image.paste(image4, (image1.width + image2.width + image3.width, 0))

        # 保存拼接后的图片到输出文件夹
        output_path = os.path.join(output_folder, file)
        result_image.save(output_path)

if __name__ == "__main__":
    folder1 = "../runs/detect/sota_fusions/docThesis_m3fd_visibleTrain_sotaFusionEval/DIVFusion"
    folder2 = "../runs/detect/sota_fusions/docThesis_m3fd_visibleTrain_sotaFusionEval/PSFusion"
    folder3 = "../runs/detect/sota_fusions/docThesis_m3fd_visibleTrain_sotaFusionEval/CDDFuse"
    folder4 = "../runs/detect/sota_fusions/docThesis_m3fd_visibleTrain_sotaFusionEval/earlyFusion"

    output_folder = "../runs/detect/sota_fusions/docThesis_m3fd_visibleTrain_sotaFusionEval/Comparison"

    horizontal_concatenate_images(folder1, folder2, folder3, folder4, output_folder)
