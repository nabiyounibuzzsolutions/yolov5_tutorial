def split_files(out_path, file_name, train=0.7, test=0.1, validate=0.2, prefix_path=''):  # split training data
    file_name = list(filter(lambda x: len(x) > 0, file_name))
    file_name = sorted(file_name)
    i, j, k = split_indices(file_name, train, test, validate)
    datasets = {'train': i, 'test': j, 'val': k}
    for key, item in datasets.items():
        if item.any():
            os.system("mkdir %s"%(out_path  + "images/" + key))
            os.system("mkdir %s"%(out_path  + "labels/" + key))
            with open(out_path + key + '.txt', 'a') as file:
                for i in item:
                    filename = str(Path(file_name[i]).name)
                    name = str(Path(file_name[i]).stem) + '.txt'
                    os.system("mv %s %s"%(out_path + "images/" + filename, out_path + "images/" + key + "/"))
                    os.system("mv %s %s"%(out_path + "labels/" + name, out_path + "labels/" + key + "/"))
                    file.write('%s%s\n' % (prefix_path, file_name[i]))

def split_indices(x, train=0.9, test=0.1, validate=0.0, shuffle=True):  # split training data
    n = len(x)
    v = np.arange(n)
    if shuffle:
        np.random.shuffle(v)

    i = round(n * train)  # train
    j = round(n * test) + i  # test
    k = round(n * validate) + j  # validate
    return v[:i], v[i:j], v[j:k]  # return indices

def convert_ath_json(json_dir):  # dir contains json annotations and images
    # Create folders
    dir = str(make_dirs(arg.output)) + "/"  # output directory

    jsons = []
    for dirpath, dirnames, filenames in os.walk(json_dir):
        for filename in [f for f in filenames if f.lower().endswith('.json')]:
            jsons.append(os.path.join(dirpath, filename))

    # Import json
    n1, n2, n3 = 0, 0, 0
    missing_images, file_name = [], []
    for json_file in sorted(jsons):
        with open(json_file) as f:
            data = json.load(f)

        # Write labels file
        for i, x in enumerate(tqdm(data['_via_img_metadata'].values(), desc='Processing %s' % json_file)):

            image_file = str(Path(json_file).parent / x['filename'])
            f = glob.glob(image_file)  # image file
            if len(f):
                f = f[0]
                file_name.append(f)
                wh = exif_size(Image.open(f))  # (width, height)
                n1 += 1  # all images
                if len(f) > 0 and wh[0] > 0 and wh[1] > 0:
                    label_file = dir + 'labels/' + Path(f).stem + '.txt'
                    nlabels = 0
                    try:
                        with open(label_file, 'a') as file:  # write labelsfile
                            for region in x['regions']:
                                box = region['shape_attributes']
                                if box["name"] == "rect":
                                    box = np.array([box['x'], box['y'], box['width'], box['height']], dtype=np.float32).ravel()
                                elif box['name'] == "polygon":
                                    x, y = min(box["all_points_x"]), min(box["all_points_y"])
                                    width, height = max(box["all_points_x"]) - x, max(box["all_points_y"]) - y
                                    box = np.array([x, y, width, height], dtype=np.float32).ravel()
                                else:
                                    print("bounding box type {0} is not supported".format(box["name"]))
                                    continue
                                category_id = int(region["region_attributes"]["class_id"])  # single-class
                                box[[0, 2]] /= wh[0]  # normalize x by width
                                box[[1, 3]] /= wh[1]  # normalize y by height
                                box = [box[0] + box[2] / 2, box[1] + box[3] / 2, box[2],
                                       box[3]]  # xywh (left-top to center x-y)

                                if box[2] > 0. and box[3] > 0.:  # if w > 0 and h > 0
                                    file.write('%g %.6f %.6f %.6f %.6f\n' % (category_id, *box))
                                    n3 += 1
                                    nlabels += 1

                        if nlabels == 0:  # remove non-labelled images from dataset
                            os.system('rm %s' % label_file)
                            continue  # next file

                        # write image
                        img_size = 4096  # resize to maximum
                        img = cv2.imread(f)  # BGR
                        assert img is not None, 'Image Not Found ' + f
                        r = img_size / max(img.shape)  # size ratio
                        if r < 1:  # downsize if necessary
                            h, w, _ = img.shape
                            img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)
                            print("Image {0} was resize!".format(Path(f).name))


                        ifile = dir + 'images/' + Path(f).name
                        if cv2.imwrite(ifile, img):  # if success append image to list
                            with open(dir + 'data.txt', 'a') as file:
                                file.write('%s\n' % ifile)
                            n2 += 1  # correct images

                    except:
                        pdb.set_trace()
                        os.system('rm %s' % label_file)
                        print('problem with %s' % f)

            else:
                missing_images.append(image_file)
    nm = len(missing_images)  # number missing
    print('\nFound %g JSONs with %g labels over %g images. Found %g images, labelled %g images successfully' %
          (len(jsons), n3, n1, n1 - nm, n2))
    if len(missing_images):
        print('WARNING, missing images:', missing_images)
    # spliting and writing images/labels to the folders
    split_files(dir, file_name, train=arg.train, test=arg.test, validate=arg.val)
    print('Done. Output saved to %s' % Path(dir).absolute())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--json', type=str, default='annotations/', help='path of the json file that includes the labels')
    parser.add_argument('--images', type=str, default='images/', help='path of the image files as the dataset')
    parser.add_argument('--output', type=str, default='output/', help='path of the output directory in which the results will be saved')
    parser.add_argument('--train', type=float, default=0.7, help='train set percentage')
    parser.add_argument('--val', type=float, default=0.2, help='val set percentage')
    parser.add_argument('--test', type=float, default=0.1, help='test set percentage')
    arg = parser.parse_args()

    convert_ath_json(json_dir=arg.json)  # images folder
