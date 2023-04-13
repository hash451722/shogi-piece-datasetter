import pathlib

import numpy as np
from PIL import Image



def disp(img_array:np.ndarray):
    img = Image.fromarray(img_array)
    img.show()


def list_imgs_path(path_dir:pathlib.Path) -> list[pathlib.Path]:
    img_path_list = list(path_dir.glob('**/*.png')) + list(path_dir.glob('**/*.jpg'))
    return img_path_list


def img2dict(img_path_list:list[pathlib.Path]) -> dict:
    d = {"em":[], "fu":[], "gi":[], "hi":[], "ka":[], "ke":[], "ki":[], "ky":[],
         "ng":[], "nk":[], "ny":[], "ou":[], "ry":[], "to":[], "um":[]}

    for i, p in enumerate(img_path_list):
        label = p.parent.name  # Directory name
        img_pil = Image.open(p)

        if img_pil.mode == "RGBA":
            img_pil = img_pil.convert("RGB")  # Remove alpha-channel

        img_ndarray = np.array(img_pil, dtype=np.uint8)


        if label in d:
            d[label].append(img_ndarray)
        else:
            print("ERROR")
            d[label] = [img_ndarray]

        if img_ndarray.shape != (64, 64, 3):
            print(p)
            print(img_ndarray.shape)
            print(img_pil.mode)
            print(type(img_pil.mode))

    return d


def savez(d:dict, path_save_dir:pathlib.Path) -> None:
    print("savez")
    # np.savez_compressed(path_save_dir.joinpath("piece"), 
    #     em=d["em"], fu=d["fu"], gi=d["gi"], hi=d["hi"], ka=d["ka"], ke=d["ke"], ki=d["ki"], ky=d["ky"],
    #     ng=d["ng"], nk=d["nk"], ny=d["ny"], ou=d["ou"], ry=d["ry"], to=d["to"], um=d["um"]
    #     )

    np.savez_compressed(path_save_dir.joinpath("em"), em=d["em"])
    np.savez_compressed(path_save_dir.joinpath("fu"), fu=d["fu"])
    np.savez_compressed(path_save_dir.joinpath("gi"), gi=d["gi"])
    np.savez_compressed(path_save_dir.joinpath("hi"), hi=d["hi"])
    np.savez_compressed(path_save_dir.joinpath("ka"), ka=d["ka"])
    np.savez_compressed(path_save_dir.joinpath("ke"), ke=d["ke"])
    np.savez_compressed(path_save_dir.joinpath("ki"), ki=d["ki"])
    np.savez_compressed(path_save_dir.joinpath("ky"), ky=d["ky"])
    np.savez_compressed(path_save_dir.joinpath("ng"), ng=d["ng"])
    np.savez_compressed(path_save_dir.joinpath("nk"), nk=d["nk"])
    np.savez_compressed(path_save_dir.joinpath("ny"), ny=d["ny"])
    np.savez_compressed(path_save_dir.joinpath("ou"), ou=d["ou"])
    np.savez_compressed(path_save_dir.joinpath("ry"), ry=d["ry"])
    np.savez_compressed(path_save_dir.joinpath("to"), to=d["to"])
    np.savez_compressed(path_save_dir.joinpath("um"), um=d["um"])
    print("END savez")


def load(path_npz:pathlib.Path):
    pieces_np = np.load(path_npz)
    for k, v in pieces_np.items():
        print(k, v.shape, type(v))




if __name__ == '__main__':
    path_current_dir = pathlib.Path(__file__).parent
    # path_img_dir = path_current_dir.parent.joinpath("piece_images", "test")
    # path_img_dir = path_current_dir.parent.joinpath("piece_images", "train_validate")
    # path_img_dir = path_current_dir.joinpath("test")
    path_img_dir = path_current_dir.joinpath("20230402")

    # Save
    path_imgs = list_imgs_path(path_img_dir)
    d = img2dict(path_imgs)

    savez(d, path_current_dir)

    # Load
    load(path_current_dir.joinpath("piece.npz"))
