import base64
import json
import pathlib
import re

import cv2
import numpy as np

import classify_piece


class CellExtractor():
    def __init__(self, cell_size:int=64,threshold:float=0.8) -> None:
        self.path_current_dir = pathlib.Path(__file__).parent
        self.cell_size = cell_size
        self.img = None
        self.points = None
        self.hv = None  # h:horizontal, v:vertical
        self.name = None
        self.piece_types = None
        self.cp = classify_piece.ClassifyPiece(threshold=threshold, bgr2rgb=True)


    def run(self, path_json:pathlib.Path) -> np.ndarray:
        dict_json = self.load_json(path_json)
        self.read_json(dict_json)
        img_extracted = self._extract_board(self.img, self.points, self.hv)
        img_cells = self._parcellate_board(img_extracted, self.hv)
        path_img_dir = self.mkdir()
        pred_labels = self.cp.run(img_cells.copy())
        self.save(img_cells, self.name, path_img_dir, pred_labels)
        return self.name, img_cells  # str ,(n, height, width, channels) (color: BGR)


    def load_json(self, path_json:pathlib.Path) -> dict:
        with open(path_json) as f:
            dict_json = json.load(f)
        return dict_json
    

    def read_json(self, dict_json:dict) -> None:
        self.name = str(pathlib.Path(dict_json["imagePath"]).stem)
        self.img = self.base64_to_cv(dict_json["imageData"])
        shapes = dict_json["shapes"]
        for shape in shapes:
            label = shape["label"]
            # if "squares-" in label:
            if re.fullmatch(r"squares-[0-9][0-9]", label):
                p = shape["points"]
                h = int(label[-2])
                v = int(label[-1])

        self.points = [tuple(p[0]), tuple(p[1]), tuple(p[3]), tuple(p[2])]
        self.hv = (h, v)

        
    def base64_to_cv(self, img_base64:str) -> np.ndarray:
        img_raw = np.frombuffer(base64.b64decode(img_base64), np.uint8)
        img_cv = cv2.imdecode(img_raw, cv2.IMREAD_UNCHANGED)
        return img_cv


    def _extract_board(self, img, corner_points, hv:tuple) -> np.ndarray:
        '''
        盤面の抽出(切り抜き)
        input: OpenCV image, [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
        return: OpenCV image (H, W, C)
        '''
        dstSize_h = self.cell_size * hv[0]
        dstSize_v = self.cell_size * hv[1]

        pts1 = np.float32(corner_points)
        pts2 = np.float32([[0,0],[dstSize_h,0],[0,dstSize_v],[dstSize_h,dstSize_v]])

        mat = cv2.getPerspectiveTransform(pts1,pts2)
        img_dst = cv2.warpPerspective(img, mat, (dstSize_h, dstSize_v))
        return img_dst



    def _parcellate_board(self, img, hv:tuple) -> np.ndarray:
        '''
        マスに画像を切り分け
        input: openCV image
        return: ndarray (n, channel, height, width)
        '''
        rows = hv[1]  # 行数　段
        cols = hv[0]  # 列数　筋
        squares = []
        for row_img in np.array_split(img, rows, axis=0):
            for chunk in np.array_split(row_img, cols, axis=1):
                squares.append(chunk)

        squares = np.array(squares)
        return squares
    

    def mkdir(self) -> pathlib.Path:
        path_img_dir = self.path_current_dir.joinpath("img_piece")
        path_img_dir.mkdir(exist_ok=True)
        self.piece_types = ["em", "fu", "gi", "hi", "ka",
             "ke", "ki", "ky", "ng", "nk",
             "ny", "ou", "ry", "to", "um"]
        for piece in self.piece_types:
            path_img_dir.joinpath(piece).mkdir(exist_ok=True)

        return path_img_dir


    def save(self, img_cells:np.ndarray, filename:str, path_img_dir, pred_labels:list[str]) -> None:
        for n, label in enumerate(pred_labels):
            # black piece turns 180 degrees.
            if label[0] == "b":
                img = np.rot90(img_cells[n], 2).copy()
            else:
                img = img_cells[n].copy()

            # character count change (3 -> 2)
            if label == "emp":
                label = "em"
            else:
                label = label[1:]

            # select Directory
            if label in self.piece_types:
                path_piece_dir = path_img_dir.joinpath(label)
            else:
                path_piece_dir = path_img_dir.joinpath()

            path_img = path_piece_dir.joinpath("{}_{:02}.png".format(filename, n))
            cv2.imwrite(str(path_img), img)



def list_json(path_dir:pathlib.Path) -> list[pathlib.Path]:
    path_json_list = list(path_dir.glob('**/*.json'))
    return path_json_list




if __name__ == '__main__':
    path_current_dir = pathlib.Path(__file__).parent
    # path_json = path_current_dir.joinpath("sample_board.json")
    # path_json2 = path_current_dir.joinpath("sample2.json")

    path_json_dir = path_current_dir.parent.joinpath("board")
    path_json_list = list_json(path_json_dir)

    # print(json_list)
    print(len(path_json_list))

    ce = CellExtractor(threshold=1.0)
    for p in path_json_list:
        print(p)
        ce.run(p)
        # break


    # path_img_dir = path_current_dir.joinpath("img_piece")
    # path_img_dir.mkdir(exist_ok=True)

    # for n, img in enumerate(img_cells):
    #     path_img = path_img_dir.joinpath("{}_{:02}.png".format(filename, n))
    #     cv2.imwrite(str(path_img), img)
