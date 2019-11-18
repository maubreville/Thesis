from fastai import *
from fastai.vision import *
from MKTreader import *
import sqlite3
testversion = 'CLE_ResNet_fastai_OC_OC'

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return out.read()

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)


def extendPolar(inpt, radius=124):
    M = radius
    pol = cv2.linearPolar(inpt, center= (round(inpt.shape[0]/2),round(inpt.shape[1]/2)),maxRadius=M, flags=cv2.WARP_FILL_OUTLIERS)
    pol2 = np.hstack([pol, cv2.flip(pol, 1)])
    extd= cv2.linearPolar(pol2, center= (round(inpt.shape[0]),round(inpt.shape[1]/2)), maxRadius=2*M, flags=(cv2.WARP_FILL_OUTLIERS | cv2.WARP_INVERSE_MAP))
    return extd[:,round(inpt.shape[0]/2):round(inpt.shape[0]*1.5)], pol, pol2


class cCLENet(nn.Module):

    def __init__(self, n_classes=2, final_bias:float=0.,  n_conv:float=4,
                 chs=256, n_anchors=9, flatten=True, sizes=None):
        super().__init__()
        self.n_classes, self.flatten = n_classes, flatten
        imsize = (256, 256)
        self.encoder = create_body(models.resnet34, True, -2)
        #self.resnet18 = models.resnet18(pretrained=True)
        #list(res18.children())[:-1]
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.finalconv = nn.Conv2d(512,2,kernel_size=(1,1),bias=True)
        self.cm = circularMask.circularMask(7,7,6.5)
        self.head = create_head(nf=1024, nc=2)
        self.flat = nn.Flatten()
    #    self.encoder = (res18.children())[:-1]
    
    def classmap(self, x):
        with torch.no_grad():
            c5 = self.encoder(x)
            cut = c5[:,:,self.cm.mask].view(-1,512,np.sum(self.cm.mask),1)
            finalconv = self.finalconv(cut).cpu()
            orig = torch.zeros((c5.shape[0],2,*c5.shape[2:4])).cpu()
            orig[:,:, learn.model.cm.mask] = finalconv[:,:,:,0].sigmoid()    
            return orig

    def forward(self, x):
        c5 = self.encoder(x)
        cut = c5[:,:,self.cm.mask].view(-1,512,29,1)
        gapped = self.gap(cut)
        finalconv = self.finalconv(gapped)
        flattened = self.flat(finalconv)
        retval=flattened
        return retval


class circularExtrapolate(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, radius=340):
        self.radius = radius

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        
        co = []
        for ci in img:
            cex = extendPolar(ci.permute(1,2,0).numpy(), radius=self.radius)
            co.append(Tensor(cex[0]).permute(2,0,1).to(img.device)[None,:,:,:])

        return torch.cat(co)
        return F.center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(radius={0})'.format(self.radius)


class MyImageItemList(ImageList):
    def open(self, fn:PathOrStr)->Image:
        filename, frame = fn.split(':')
        reader = MKTreader(filename, verbose=0)
        arr = reader.readImageUINT8(int(frame))
        arr_rgb = np.zeros((arr.shape[0], arr.shape[1],3), dtype=np.uint8)
        for k in range(3):
            arr_rgb[:,:,k] = arr
        image = PIL.Image.fromarray(arr_rgb)
        tensor = pil2tensor(image, np.float32)
        tensor = circularExtrapolate(radius=arr.shape[0]*0.47)(tensor[None,:,:,:])
        image = Image(tensor[0]/255.0)
        image = image.resize((3,224,224))
#        image = self.__super__.open(self, fn)
        return image

import sqlite3
conn = sqlite3.connect('../CLEdB.db')
DB = conn.cursor()

patients_train = [1,2,3,4,5,6,7,8,9,10,11,12]
uidclause = '(%s)' % (','.join([str(x) for x in patients_train]))
print('Patients to use: ', uidclause)

for testPatient in patients_train:
    print('Testing on patient: ',testPatient)
    valpatients = DB.execute(f'SELECT patientUID FROM CLEsequences  where patientUID in {uidclause} and patientUID != {testPatient} group by patientUID ORDER BY RANDOM() LIMIT 2').fetchall()
    valpatient_str = '(%s)' % (','.join([str(x[0]) for x in valpatients]))
    print('Validation patients:', valpatient_str)

    res = DB.execute(f'SELECT  CLEdatabases.path, CLEsequences.subfolder, CLEsequences.patient, CLEsequences.file, frameIdx, cellStructure==1, patientUID in {valpatient_str} FROM CLEframes LEFT JOIN CLEsequences on CLEsequences.id = CLEframes.sequenceId LEFT JOIN CLEdatabases on CLEdatabases.id == CLEsequences.database where patientUID in {uidclause} AND cellStructure in (0,1,3) AND CLEframes.imageQuality>0 and CLEframes.motionArtifactClass<2 and CLEframes.gaussianNoiseClass<=2 and patientUID!={testPatient}').fetchall()
    resname = ['%s/%s/%s%s.mkt:%d' % (x[0], x[2],(x[1]+'/') if x[1] is not None else '',x[3],x[4]) for x in res]
    df = pd.DataFrame({'File':resname, 'Class': [x[5] for x in res], 'IsVal':[x[6] for x in res]}, columns=['File','Class','IsVal'])

    res = DB.execute(f'SELECT  CLEdatabases.path, CLEsequences.subfolder, CLEsequences.patient, CLEsequences.file, frameIdx, cellStructure==1, patientUID=={testPatient}, CLEframes.id FROM CLEframes LEFT JOIN CLEsequences on CLEsequences.id = CLEframes.sequenceId LEFT JOIN CLEdatabases on CLEdatabases.id == CLEsequences.database where patientUID in {uidclause} AND cellStructure in (0,1,3) AND CLEframes.imageQuality>0 and CLEframes.motionArtifactClass<2 and CLEframes.gaussianNoiseClass<=2').fetchall()
    resname = ['%s/%s/%s%s.mkt:%d' % (x[0], x[2],(x[1]+'/') if x[1] is not None else '',x[3],x[4]) for x in res]
    df_test = pd.DataFrame({'File':resname, 'Class': [x[5] for x in res], 'IsVal':[x[6] for x in res], 'Frame':[x[7] for x in res]}, columns=['File','Class','IsVal','Frame'])



    tfms = get_transforms(max_rotate=180)

    # Training data set
    data = MyImageItemList.from_df(df, path=Path('../Data'))
    data_split = data.split_from_df('IsVal')
    data_split = data_split.label_from_df('Class')
    data_split = data_split.transform(tfms)
    data = data_split.databunch(bs=16)
    print('Training data:',df.head(10))

    # Test data set
    data_test = MyImageItemList.from_df(df_test, path=Path('../Data'))
    data_test = data_test.split_from_df('IsVal')
    data_test = data_test.label_from_df('Class')
    data_test = data_test.transform(get_transforms(max_rotate=180))
    data_test = data_test.databunch(bs=16)


    model = cCLENet().cuda()
    learn=Learner(data,model)
    learn.model_dir='/tmp/'

    from fastai.callbacks import SaveModelCallback
    learn.metrics = [accuracy]
    learn.fit(10,1e-4, callbacks=[ShowGraph(learn),SaveModelCallback(learn, every='improvement', monitor='accuracy')])

    conn = sqlite3.connect('results/CLEresults_'+testversion+'.db',detect_types=sqlite3.PARSE_DECLTYPES)
    cur = conn.cursor()
    cur.execute('CREATE TABLE if not exists "CNNresults" ( '
    '`frameUID`	INTEGER,'
    '`coord_x1`	INTEGER,'
    '`coord_x2`	INTEGER,'
    '`coord_y1`	INTEGER,'
    '`coord_y2`	INTEGER,'
    '`result`	INTEGER,'
    '`classmap`  ARRAY,'
    '`classmap_noise` ARRAY,'
    '`correct_result`	INTEGER,'
    '`sequenceId` INTEGER,'
    '`id`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,'
    '`patientId`	INTEGER,'
    '`prob_0`	REAL,'
    '`prob_1`	REAL)')
    cur.execute('DELETE from CNNresults WHERE patientId=='+str(testPatient))
    conn.commit()

    learn.model.eval()
    idx_to_frame = np.array(df_test.Frame[df_test.IsVal==1])

    idx=0
    with torch.no_grad():
        for b,label in iter(data_test.valid_dl):
            classmap_total = learn.model.classmap(b.cuda()).numpy()
            score = learn.model(b.cuda()).sigmoid()
            for batchI in range(score.shape[0]):
                print(f'Scores are: {score[batchI]}   Label: {label[batchI]}  Frame: {idx_to_frame[idx]}')
                cur.execute('INSERT INTO CNNresults (frameUID, coord_x1, coord_y1, '
                        'coord_x2, coord_y2, correct_result, result, prob_0, prob_1, patientId, classmap) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                        (int(idx_to_frame[idx]),0,0,576,576,int(label[batchI]),
                        int(score[batchI,1]>0.5),float(score[batchI,0]),
                        float(score[batchI,1]), testPatient,np.squeeze(classmap_total[batchI]  ).reshape(-1).tobytes()))
                idx+=1
            
                
            
    conn.commit()
