import numpy as np
from matplotlib import pyplot as plt
import time

micData = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3338, 7021, 9332, 11582, 12461, 13400, 13929, 14206, 14427, 14351, 14253, 13931, 13590, 13141, 12714, 12252, 11781, 11329, 10886, 10472, 10054, 9622, 9189, 8724, 8242, 7737, 7187, 6603, 5959, 5256, 4498, 3726, 2886, 2015, 1078, 127, -877, -1899, -2938, -3995, -5004, -5989, -6926, -7853, -8773, -9660, -10528, -11378, -12178, -12988, -13751, -14486, -15238, -15937, -16630, -17266, -17849, -18422, -18938, -19440, -19875, -20237, -20530, -20794, -7353, 681, 8839, 11830, 13549, 13558, 13402, 12320, 11269, 9641, 8054, 6194, 4428, 2662, 1101, -305, -1502, -2468, -3205, -3724, -4067, -4281, -4378, -4416, -4417, -4402, -4428, -4488, -4627, -4856, -5158, -5512, -5947, -6432, -6991, -7556, -8175, -8786, -9401, -10004, -10534, -11022, -11434, -11830, -12204, -12548, -12883, -13210, -13518, -13856, -14191, -14534, -14934, -15334, -15760, -16183, -16587, -17022, -17439, -17864, -18261, -18597, -18894, -19174, -19455, -19734, -19959, -20172, -20311, -20377, -20407, -20369, -20300, -20173, -19966, -19711, -19391, -19014, -18558, -18063, -17487, -16844, -16138, -15336, -14488, -13546, -12540, -11450, -10282, -9081, -7830, -6575, -5278, -3956, -2632, -1318, 40, 1393, 2712, 4057, 5338, 6608, 7862, 9081, 10285, 11430, 12522, 13523, 14454, 15335, 16137, 16911, 17599, 18241, 18865, 19430, 19928, 20354, 20749, 21077, 21319, 21509, 21625, 21626, 21533, 21354, 21080, 20755, 20353, 19851, 19303, 18686, 18000, 17252, 16442, 15525, 14560, 13540, 12453, 11350, 10211, 9064, 7899, 6730, 5550, 4346, 3163, 1973, 808, -340, -1495, -2606, -3721, -4811, -5867, -6899, -7848, -8757, -9622, -10410, -11136, -11782, -12354, -12875, -13371, -13802, -14186, -14522, -14780, -14970, -15066, -15077, -15028, -14907, -14711, -14451, -14091, -13648, -13115, -12527, -11894, -11173, -10366, -9474, -8511, -7481, -6393, -5218, -4005, -2752, -1471, -195, 1117, 2429, 3751, 5063, 6356, 7650, 8936, 10180, 11400, 12579, 13744, 14838, 15906, 16935, 17878, 18768, 19573, 20329, 21011, 21601, 22141, 22643, 23133, 23528, 23878, 24157, 24370, 24522, 24614, 24619, 24509, 24366, 24118, 23798, 23391, 22917, 22392, 21785, 21129, 20389, 19594, 18722, 17806, 16836, 15781, 14658, 13493, 12279, 11063, 9842, 8584, 7340, 6061, 4796, 3496, 2229, 984, -269, -1496, -2709, -3910, -5072, -6210, -7299, -8331, -9313, -10249, -11116, -11916, -12661, -13318, -13906, -14454, -14969, -15454, -15849, -16173, -16439, -16601, -16701, -16750, -16704, -16569, -16368, -16070, -15704, -15269, -14764, -14155, -13477, -12733, -11898, -11004, -10034, -8977, -7859, -6660, -5429, -4174, -2907, -1631, -308, 997, 2291, 3593, 4850, 6129, 7389, 8580, 9772, 10929, 12033, 13117, 14148, 15113, 15999, 16800, 17543, 18239, 18848, 19393, 19894, 20358, 20769, 21079, 21350, 21542, 21699, 21778, 21761, 21680, 21491, 21268, 20939, 20512, 20022, 19442, 18796, 18088, 17301, 16487, 15614, 14670, 13654, 12579, 11459, 10281, 9074, 7842, 6595, 5344, 4099, 2827, 1551, 261, -1011, -2274, -3522, -4750, -5974, -7168, -8345, -9499, -10608, -11666, -12666, -13620, -14508, -15335, -16080, -16729, -17332, -17885, -18409, -18897, -19306, -19645, -19907, -20090, -20210, -20260, -20208, -20075, -19848, -19543, -19161, -18703, -18170, -17554, -16877, -16130, -15305, -14398, -13419, -12381, -11274, -10085, -8869, -7627, -6366, -5085, -3778, -2477, -1183, 89, 1381, 2682, 3948, 5188, 6392, 7559, 8698, 9784, 10847, 11859, 12777, 13630, 14404, 15127, 15778, 16348, 16873, 17344, 17786, 18159, 18454, 18690, 18890, 19004, 19037, 18995, 18878, 18659, 18374, 18004, 17559, 17062, 16468, 15809, 15090, 14307, 13489, 12589, 11640, 10641, 9549, 8405, 7226, 6015, 4799, 3569, 2319, 1073, -188, -1464, -2731, -3983, -5209, -6450, -7631, -8813, -9999, -11127, -12233, -13267, -14235, -15171, -16055, -16868, -17598, -18242, -18806, -19346, -19839, -20305, -20737, -21069, -21343, -21483, -21554, -21585, -21531, -21454, -21280, -21011, -20626, -20179, -19648, -19060, -18411, -17667, -16848, -15950, -15013, -13960, -12836, -11671, -10441, -9198, -7916, -6591, -5256, -3909, -2575, -1276, 38, 1353, 2650, 3939, 5202, 6408, 7577, 8688, 9772, 10799, 11758, 12626, 13442, 14199, 14867, 15494, 16055, 16582, 17043, 17437, 17763, 18041, 18280, 18449, 18531, 18543, 18471, 18292, 18029, 17707, 17332, 16880, 16370, 15775, 15079, 14361, 13573, 12712, 11817, 10847, 9807, 8696, 7531, 6346, 5143, 3924, 2699, 1475, 217, -1026, -2268, -3521, -4730, -5933, -7100, -8264, -9426, -10528, -11606, -12652, -13635, -14588, -15483, -16310, -17079, -17754, -18346, -18865, -19329, -19793, -20210, -20556, -20840, -21025, -21129, -21170, -21130, -21032, -20855, -20583, -20239, -19823, -19324, -18749, -18081, -17345, -16518, -15621, -14661, -13616, -12509, -11344, -10111, -8834, -7535, -6217, -4895, -3587, -2261, -939, 374, 1707, 3012, 4298, 5570, 6778, 7937, 9053, 10138, 11198, 12184, 13086, 13894, 14643, 15356, 15992, 16550, 17084, 17549, 17966, 18322, 18613, 18861, 19047, 19173, 19196, 19153, 19013, 18768, 18479, 18083, 17639, 17133, 16542, 15896, 15171, 14392, 13564, 12686, 11716, 10689, 9607, 8482, 7303, 6098, 4903, 3671, 2457, 1246, 11, -1234, -2452, -3673, -4865, -6021, -7193, -8338, -9445, -10536, -11595, -12594, -13540, -14429, -15255, -16024, -16716, -17308, -17846, -18352, -18825, -19261, -19620, -19906, -20132, -20261, -20326, -20306, -20214, -20044, -19784, -19452, -19054, -18581, -18022, -17386, -16670, -15874, -14985, -14041, -13018, -11946, -10792, -9568, -8314, -7035, -5714, -4399, -3053, -1711, -399, 894, 2205, 3526, 4814, 6074, 7290, 8455]
perfectData = [0.0, 0.05701506851315605, 0.11384464624742195, 0.17030384589295444, 0.22620898511469945, 0.28137818413791266, 0.3356319574692951, 0.3887937978286591, 0.4406907503913828, 0.4911539754734333, 0.5400192978283382, 0.5871277407690446, 0.6323260433769697, 0.6754671591155771, 0.7164107342263037, 0.7550235643504442, 0.7911800278914337, 0.8247624947076444, 0.8556617088060686, 0.8837771437918485, 0.9090173299172427, 0.9313001516660233, 0.9505531149051507, 0.9667135827345862, 0.9797289792679352, 0.9895569606809512, 0.9961655529714141, 0.9995332559822022, 0.999649113349133, 0.9965127481460041, 0.9901343641108727, 0.9805347124495796, 0.9677450243245203, 0.9518069092483014, 0.9327722197128454, 0.9107028824943529, 0.8856706971829496, 0.8577571025924757, 0.8270529118103702, 0.7936580167496288, 0.7576810631640374, 0.7192390971839737, 0.678457184522725, 0.6354680035921814, 0.5904114138516521, 0.5434340007941123, 0.4946885990502237, 0.44433379516162474, 0.3925334116411645, 0.339455973998619, 0.285274162465838, 0.2301642502050746, 0.17430552982820147, 0.11787973009255745, 0.0610704246711291, 0.004062434920542901, -0.05295877141010324, -0.10980768357336257, -0.16629935135745583, -0.22224998679665012, -0.27747756210043917, -0.33180240185622795, -0.38504776757887127, -0.43704043270532683, -0.48761124616373974, -0.536595682683473, -0.5838343780557221, -0.6291736476033155, -0.6724659861729316, -0.7135705480230723, -0.7523536050465491, -0.7886889818367089, -0.8224584661819753, -0.85355219365322, -0.8818690050327548, -0.9073167754221019, -0.9298127139578345, -0.9492836331604025, -0.9656661870396532, -0.9789070771824036, -0.988963226151586, -0.9958019176328372, -0.9994009028725814, -0.9997484730613277, -0.9968434974266908, -0.9906954269122047, -0.981324263429963, -0.9687604947871153, -0.9530449954979279, -0.9342288938041063, -0.9123734053360071, -0.8875496339559025, -0.8598383404312238, -0.829329679690379, -0.7961229075159397, -0.7603260576294414, -0.7220555902183479, -0.681436013048653, -0.6385994763957817, -0.5936853431116248, -0.5468397352264313, -0.49821505856064363, -0.44796950689326076, -0.39626654729987504, -0.3432743883347513, -0.2891654327871485, -0.23411571679227258, -0.17830433712162375, -0.12191286851598397, -0.0651247729566724, -0.008124802796912191, 0.0489016003048488, 0.10576890869417463, 0.16229211230966217, 0.21828732058803899, 0.2735723607271216, 0.32796737036028045, 0.3812953827142079, 0.4333829023462761, 0.48406046958838306, 0.5331632118609626, 0.5805313800635171, 0.6260108682966115, 0.6694537152244824, 0.7107185854471383, 0.7496712293158803, 0.7861849196962969, 0.8201408642577721, 0.8514285919481983, 0.8799463123965325, 0.9056012470739389, 0.9283099311361266, 0.9479984849648795, 0.9646028545253592, 0.9780690197572011, 0.9883531703214429, 0.9954218481315098, 0.999252056204553, 0.9998313334790092, 0.9971577953549713, 0.9912401398254777, 0.9820976191787739, 0.9697599773636097, 0.954267353221341, 0.9356701498996637, 0.9140288708728233, 0.8894139231017769, 0.8619053879747174, 0.8315927607731548, 0.7985746595111757, 0.7629585040951347, 0.724860166847588, 0.6844035955324275, 0.6417204101076787, 0.596949474517848, 0.5502364449189426, 0.5017332958059602, 0.45159782558452033, 0.39999314319518925, 0.3470871374606978, 0.2930519308824788, 0.2380633196634829, 0.1823002017791226, 0.12594399495701877, 0.06917804645906965, 0.01218703658603983, -0.04484362215469389, -0.10172838826355769, -0.15828219488282633, -0.21432105188651424, -0.26966264446723803, -0.3241269262726856, -0.3775367051619484, -0.4297182196760831, -0.48050170434741213, -0.5297219420084041, -0.5772188013032759, -0.622837757653613, -0.6664303959830786, -0.707854893565709, -0.746976481426881, -0.7836678827958512, -0.8178097271834526, -0.8492909387377487, -0.8780090976142356, -0.9038707731848825, -0.9267918280019929, -0.9466976915279551, -0.9635236027403543, -0.9772148208231629, -0.987726803258545, -0.9950253507398881, -0.9990867184345996, -0.9998976932346948, -0.9974556367438469, -0.9917684938610517, -0.982854766932978, -0.9707434555590977, -0.9554739622454269, -0.937095964213827, -0.9156692517839146, -0.8912635338533778, -0.8639582111095512, -0.833842117710063, -0.8010132322729806, -0.7655783591166806, -0.7276527807865143, -0.6873598829987126, -0.6448307532217147, -0.600203754200909, -0.5536240738142291, -0.5052432527231356, -0.4552186913556437, -0.4037131378254336, -0.3508941584529632, -0.29693359261121544, -0.2420069936696513, -0.18629305785516231, -0.12997304288818914, -0.07323017828534312, -0.01624906924707423, 0.040784903930263156, 0.09768618896402252, 0.15426966525440264, 0.21035124614917416, 0.2657484778445789, 0.32028113297400285, 0.3737717969532213, 0.42604644517463847, 0.47693500917271575, 0.5262719299186043, 0.5738966964439656, 0.61965436804158, 0.6633960783439067, 0.7049795196395722, 0.7442694058521732, 0.7811379126751536, 0.8154650934308092, 0.8471392693005143, 0.8760573926565813, 0.9021253823137141, 0.9252584296093684, 0.9453812743172033, 0.9624284494960126, 0.9763444944775147, 0.9870841353001067, 0.9946124320015448, 0.9989048922913643, 0.99994755123322, 0.9977370166779085, 0.9922804802992647, 0.9835956941970281, 0.9717109131427996, 0.9566648026569778, 0.9385063132157462, 0.9172945209973448, 0.893098435685756, 0.865996775957069, 0.8360777133789649, 0.8034385855565032, 0.7681855794574426, 0.7304333859473768, 0.6903048266585762, 0.6479304544065089, 0.6034481284539777, 0.5570025660047477, 0.5087448713857895, 0.45883204444987136, 0.4074264697978818, 0.3546953884825821, 0.3008103539125689, 0.2459466737265727, 0.19028283945386076, 0.13399994581631824, 0.07728110156135128, 0.0203108337424769, -0.036725512614388375, -0.09364237750578264, -0.1502545896449529, -0.20637796889149002, -0.2618299254563774, -0.31643005393307355, -0.37000072022198915, -0.42236763943887534, -0.4733604429270529, -0.5228132325286547, -0.5705651203117555, -0.6164607519974076, -0.6603508123836614, -0.702092511122298, -0.7415500472678331, -0.778595051087427, -0.8131070016943794, -0.8449736191464567, -0.8740912297334289, -0.900365103265398, -0.9237097612646071, -0.9440492550580437, -0.9613174128661338, -0.9754580550836396, -0.9864251770523642, -0.994183098731062, -0.9987065807756048, -0.999980906651756, -0.9980019305134175, -0.992776090690576, -0.9843203887430716, -0.9726623341483325, -0.9578398548030261, -0.9399011736298039, -0.9189046516905733, -0.894918598316708, -0.8680210488739263, -0.8382995108848268, -0.8058506793350544, -0.7707801220893039, -0.7332019364406073, -0.6932383779103055, -0.6510194625063125, -0.6066825437337027, -0.5603718657337332, -0.5122380940051454, -0.46243782523442717, -0.4111330778297568, -0.35849076481616, -0.30468215080661654, -0.24988229481594654, -0.19426948073007802, -0.13802463728363634, -0.08133074943290942, -0.024372263039140328, 0.032665515201014954, 0.08959702062566212, 0.14623703431706236, 0.20240128568623023, 0.2579070519722548, 0.3125737527059727, 0.36622353720400336, 0.4186818631817622, 0.4697780646030831, 0.519345906918972, 0.5672241278891388, 0.6132569622267672, 0.6572946483597213, 0.6991939156594832, 0.7388184505526483, 0.7760393399986512, 0.8107354908908009, 0.8427940240162608, 0.8721106412932367, 0.8985899650906028, 0.9221458485260748, 0.9427016557333822, 0.9601905111866529, 0.9745555172708424, 0.9857499393903983, 0.9937373580139186, 0.9984917871601445, 0.9999977589398233, 0.9982503738783829, 0.9932553168557072, 0.9850288386111515, 0.9735977028739711, 0.958999099291161, 0.941280522435997, 0.9204996172908917, 0.8967239917072782, 0.8700309964526458, 0.8405074735603357, 0.8082494738007812, 0.7733619441933692, 0.7359583865755845, 0.6961604883401962, 0.6540977265418514, 0.6099069466610846, 0.5637319173961296, 0.5157228629309989, 0.4660359742014994, 0.4148329007492627, 0.3622802248169084, 0.3085489193953708, 0.2538137919864696, 0.19825291589048627, 0.14204705086886332, 0.08537905506686899, 0.028433290109487155, -0.02860497869409103, -0.08555018508598981, -0.142217065574235, -0.19842126216236786, -0.2539799221331315, -0.30871229293495606, -0.3624403102358064, -0.4149891772313038, -0.4661879333223888, -0.515870010312383, -0.5638737743139959, -0.6100430516032246, -0.654227636709328, -0.6962837810879303, -0.7360746607873753, -0.773470821586864, -0.8083506001581731, -0.8406005198807627, -0.8701156600225426, -0.8967999970852251, -0.9205667172037195, -0.9413384985832535, -0.9590477630553295, -0.9736368959341066, -0.9850584334579581, -0.9932752172063717, -0.9982605149898184, -0.9999981078193014, -0.9984823426726335, -0.9937181508857753, -0.9857210321094046, -0.9745170038829095, -0.9601425169898556, -0.9426443368703189, -0.922079391475876, -0.898514586062256, -0.8720265855221673, -0.8427015649664997, -0.8106349293653022, -0.7759310031606792, -0.7387026908613838, -0.6990711097233512, -0.6571651957111572, -0.6131212840223669, -0.5670826655395086, -0.5191991206526493, -0.4696264319692357, -0.41852587749656545, -0.36606370594569054, -0.31241059586383024, -0.257741100354894, -0.2022330791946941, -0.14606712018832077, -0.0894259516522436, -0.032493847932581706, 0.02454397010646162, 0.08150193767349406, 0.13819474975980614, 0.19443796400400457, 0.2500486007501896, 0.30484573834741463, 0.3586511017536755, 0.4112896425295513, 0.46259010833451253, 0.5123856000731554, 0.5605141148787135, 0.6068190731673846, 0.6511498280487391, 0.6933621554348942, 0.7333187232539968, 0.770889538241467, 0.8059523688554082, 0.838393142940331, 0.868106318845408, 0.8949952287899025, 0.9189723933586491, 0.9399598061044473, 0.9578891873314449, 0.9727022062338494, 0.9843506706672741, 0.9927966839353372, 0.9980127680814137, 0.9999819532844324, 0.9986978330678844, 0.9941645851424235, 0.986396957814252, 0.9754202220035112, 0.9612700890287763, 0.94399259442513, 0.923643948173809, 0.9002903518306662, 0.8740077831483967, 0.8448817488932525, 0.8130070066603918, 0.7784872565929111, 0.7414348040075309, 0.7019701940244967, 0.660221819390422, 0.6163255027699076, 0.5704240548649806, 0.5226668097998666, 0.4732091392827115, 0.4222119471248273, 0.3698411457620214, 0.3162671164810204, 0.26166415510709284, 0.20620990495628258, 0.15008477889700647, 0.0934713724012886, 0.03655386949522185, -0.020482556458750266, -0.07745234519820215, -0.13417015325584242, -0.19045145694928123, -0.24611315270377085, -0.3009741527548185, -0.3548559742926065, -0.4075833201315648, -0.45898464901596003, -0.5088927337060636, -0.5571452050292396, -0.6035850801259997, -0.648061273172402, -0.6904290869172625, -0.7305506834349851, -0.7682955325625338, -0.8035408365615904, -0.8361719296242967, -0.8660826509228893, -0.8931756899895337, -0.917362903302699, -0.9385656010501364, -0.9567148031354897, -0.9717514635956704, -0.9836266626988693, -0.9923017660982622, -0.9977485505236081, -0.9999492956018218, -0.9988968415078017, -0.9945946122579484, -0.9870566045705912, -0.9763073423295676, -0.9623817967991002, -0.945325272849536, -0.9251932615641117, -0.9020512597062506, -0.8759745566347434, -0.8470479893600437, -0.8153656665385891, -0.7810306623030712, -0.7441546809247408, -0.7048576933987526, -0.663267547134826, -0.6195195500230599, -0.5737560302281116, -0.5261258731438407, -0.47678403701491495, -0.4258910488011989, -0.3736124819251319, -0.3201184176010634, -0.2655828914991491, -0.21018332754392766, -0.15409996068970852, -0.0975152505506264, -0.04061328779306623, 0.01642080477828672, 0.07340147449235017, 0.13014334248206022, 0.18646180678930702, 0.24217364294233815, 0.29709760005167635, 0.35105499048527566, 0.40387027120443414, 0.4553716148692298, 0.5053914688554348, 0.5537671003641914, 0.6003411258510869, 0.6449620230521115, 0.6874846239407743, 0.7277705870125248, 0.7656888473600938, 0.8011160430753094, 0.8339369165903306, 0.8640446896524796, 0.8913414107127782, 0.915738273598002, 0.9371559064295104, 0.9555246298488518, 0.9707846837101008, 0.9828864215013716, 0.9917904718629977, 0.9974678666769022, 0.9999001353104339, 0.999079364708061, 0.9950082251354216, 0.9876999614919784, 0.9771783502205366, 0.9634776219538187, 0.9466423501497514, 0.9267273060777763, 0.9037972806279664, 0.8779268735226684, 0.8492002506164416, 0.8177108700738895, 0.7835611783162038, 0.746862276725675, 0.7077335601924253, 0.6663023286793592, 0.6227033730690306, 0.5770785366398189, 0.5295762535981021, 0.48035106616770956, 0.429563121807829, 0.3773776521949843, 0.3239644356642308, 0.26949724485841, 0.2141532813824729, 0.15811259930210686, 0.1015575193623376, 0.04467203583172237, -0.012358782097961672, -0.06934939240925882, -0.12611438389470847, -0.1824690793670527, -0.2382301364813661, -0.29321614421446485, -0.3472482130610224, -0.40015055702626345, -0.4517510655218405, -0.501881863304209, -0.5503798566339502, -0.5970872638790858, -0.6418521288361813, -0.6845288150992064, -0.7249784798677912, -0.7630695256534321, -0.7986780284140105, -0.8316881407238472, -0.8619924686675566, -0.8894924212315594, -0.914098531056549, -0.9357307455073903, -0.9543186871134909, -0.9698018825323399, -0.98212995929131, -0.9912628096676586, -0.9971707211735459, -0.9998344732215829, -0.9992453996563997, -0.9954054169488045, -0.9883270179608059, -0.978033231301786, -0.9645575464080419, -0.9479438045894636, -0.9282460563977839, -0.9055283857804526, -0.8798647015922211, -0.8513384971427251, -0.8200425785623497, -0.7860787628700778, -0.749557546725671, -0.7105977469438035, -0.6693261139396802, -0.6258769193637802, -0.5803915192673103, -0.5330178942194786, -0.48391016787282387, -0.4332281055428512, -0.3811365944332926, -0.3278051071979661, -0.2734071505845612, -0.21811970095398378, -0.16212262851181738, -0.10559811212504824, -0.04873004662787356, 0.008296555455148905, 0.06529616582224067, 0.12208334398548243, 0.17847334057627454, 0.23428269840229898, 0.2893298493005689, 0.34343570484477726, 0.39642423898515977, 0.4481230607253313, 0.4983639749729704, 0.5469835297396997, 0.5938235479099165, 0.6387316418485703, 0.6815617091735943, 0.7221744080801772, 0.7604376106703887, 0.7962268328133423, 0.8294256391374022, 0.8599260218368414, 0.8876287520605778, 0.9124437027397361, 0.9342901418038366, 0.9530969948316144, 0.9688030762819908, 0.9813572885529179, 0.99071878822049, 0.9968571189174645, 0.9997523104189195, 0.9993949436126698, 0.9957861811430646, 0.9889377636284838, 0.978871971464831, 0.9656215523392944, 0.9492296146901869, 0.9297494874595198, 0.9072445465945077, 0.8817880088625547, 0.8534626936504556, 0.8223607535227377, 0.7885833744158672, 0.7522404464434944, 0.7134502063839445, 0.6723388530129029, 0.6290401365328546, 0.5836949234349457, 0.5364507382090536, 0.4874612833928259, 0.43688593952141863, 0.3848892466045517, 0.3316403688179557, 0.2773125441506763, 0.22208252079888818, 0.1661299821395589, 0.10963696215506971, 0.05278725321035236, -0.004234191890584726, -0.06124186162351073, -0.11805028928044042, -0.17447465636044063, -0.2303313938514766, -0.2854387794472519, -0.3396175287560755, -0.39269137857819697, -0.44448766035427234, -0.4948378619189883, -0.5435781757325301, -0.5905500318061433, -0.635600613588077, -0.678583355131409, -0.7193584179265241, -0.7577931458466202, -0.7937624967264695, -0.8271494491700556, -0.8578453832638303, -0.8857504339567961, -0.9107738159579379, -0.9328341190937769, -0.9518595731653563, -0.9677882814428015, -0.9805684220379265, -0.9901584164997222, -0.996527065084175, -0.9996536482584131, -0.9995279941088805, -0.9961505114342798, -0.989532188415603, -0.9796945568675705, -0.9666696221878104, -0.9504997592316284, -0.9312375744511889, -0.9089457347475622, -0.8836967635924824, -0.8555728050830678, -0.8246653566971995, -0.7910749716188602, -0.754910931602051, -0.7162908914374344, -0.6753404961784572, -0.6321929723722661, -0.5869886946251858, -0.5398747289130612, -0.49100435412206156, -0.44053656337666075, -0.3886355467770772, -0.3354701572291615, -0.28121336110431866, -0.2260416755169906, -0.17013459405017486, -0.1136740027974533, -0.056843588621263694, 0.00017175844727223194, 0.05718654672304351, 0.11401528633886714, 0.17047309271160435, 0.22637628803901785, 0.2815429988705802, 0.3357937478079584, 0.38895203741044676, 0.44084492440530854, 0.4913035823352838, 0.5401638508125276, 0.5872667695920668, 0.6324590957274446, 0.6755938021257598, 0.7165305558803672, 0.7551361748249124, 0.7912850608234371, 0.8248596083868006, 0.8557505872862251, 0.8838574979189383, 0.9090888982700346, 0.9313627014066026, 0.95060644253644, 0.966757514762378, 0.9797633727653499, 0.9895817037534147, 0.9961805651207033, 0.999538488368329, 0.9996445489492399, 0.996498401809746, 0.9901102825121053, 0.9805009739345124, 0.9677017386568297, 0.9517542172520227, 0.9327102928142336, 0.9106319221641532, 0.885590934280961, 0.8576687966164598, 0.8269563500518217, 0.7935535133591094, 0.757568958129134, 0.7191197552231768, 0.678330993898899, 0.6353353748493711, 0.5902727784794513, 0.5432898098238691, 0.49453932158767455, 0.44417991686069924, 0.3923754331240212, 0.33929440922689225, 0.28510953706855435, 0.22999709976861044, 0.17413639815377757, 0.11770916742711139, 0.06089898591711104, 0.003890677830646448, -0.05313028804752194, -0.10997840175222112, -0.16646871566936758, -0.22241744623782392, -0.277642571864343, -0.3319644251060135, -0.38520627719390954, -0.43719491299611524, -0.4877611945496537, -0.5367406113277999, -0.5839738154528206, -0.6293071401125476, -0.6725930994945614, -0.7136908686111838, -0.7524667414544466, -0.7887945659904652, -0.8225561545778919, -0.8536416684753735, -0.8819499751869646, -0.9073889774829791, -0.929875913025778, -0.9493376236258366, -0.9657107932519295, -0.9789421540212733, -0.9889886594993186, -0.9958176247454932, -0.9994068326492026, -0.9997446062101919, -0.9968298465280728, -0.9906720363774484, -0.9812912093569952, -0.968717884712871, -0.9529929680484965, -0.9341676182437214, -0.9123030810163812, -0.8874704896676577, -0.8597506336595422, -0.8292336957773323, -0.7960189587321403, -0.7602144821581431, -0.7219367510551824, -0.681310296820691, -0.6384672921036899, -0.5935471207990429, -0.5466959245808634, -0.49806612745049683, -0.4478159398456609, -0.3961088439243467, -0.34311306169779987, -0.28900100774307586, -0.23394872827560503, -0.1781353284068248, -0.1217423894499383, -0.06495337816985876, -0.007953049898982408, 0.04907315253918323, 0.10593970214301444, 0.16246159131973487, 0.21845493378241115, 0.2737375627990386, 0.32812962384723937, 0.38145415974654817, 0.4335376863644826, 0.48421075702368993, 0.5333085137736175, 0.5806712237334883, 0.6261447987615142, 0.6695812967597569, 0.7108394029835832, 0.7497848897900665, 0.7862910533293038, 0.8202391257582501, 0.8515186616357023, 0.8800278972415793, 0.9056740816513128, 0.9283737784884317, 0.9480531373734276, 0.9646481341859621, 0.9781047793586387, 0.9883792935247075, 0.995438249948311, 0.9992586832738068, 0.999828164240447, 0.9971448401192803, 0.9912174407407559, 0.9820652500934116, 0.9697180435860249, 0.9542159911773849, 0.9356095266887665, 0.9139591837243641, 0.8893353987334259, 0.8618182818548416, 0.831497356289672, 0.7984712670496181, 0.7628474600288255, 0.7247418324433036, 0.6842783557750851, 0.6415886724478025, 0.5968116675460249, 0.5500930169714263, 0.5015847135060921, 0.45144457232462554, 0.39983571756393155, 0.3469260516209679, 0.2928877089051815, 0.2378964958225022, 0.18213131881316075, 0.12577360230385976, 0.06900669846806497, 0.012015290714585099, -0.04501520715472821, -0.1018992541636922, -0.15845178579407188, -0.2144888160678813, -0.2698280361207512, -0.3242894073190834, -0.37769574699121566, -0.42987330486723846, -0.4806523283518571, -0.5298676147913977, -0.5773590489382243, -0.6229721238638781, -0.6665584436264583, -0.7079762060565933, -0.7470906640915604, -0.7837745641565381, -0.8179085601668422, -0.8493816018041506, -0.8780912958036876, -0.9039442390767376, -0.9268563225849566, -0.9467530049776617, -0.9635695551020153, -0.9772512625970088, -0.9877536158862189, -0.9950424469901465, -0.9990940426871165, -0.9998952216610094, -0.9974433773848884, -0.9917464866009785, -0.982823083369421, -0.9707021987702256, -0.9554232664545986, -0.9370359943529065, -0.9156002029567012, -0.8911856307008437, -0.8638717070790239, -0.8337472942306507, -0.8009103978399887, -0.765467848287965, -0.7275349530940473, -0.6872351217788717, -0.6446994643681889, -0.6000663648441389, -0.5534810309318228, -0.5050950216856732, -0.45506575441267355, -0.4035559925365062, -0.35073331606893127, -0.29676957641093055, -0.24184033725752127, -0.1861243034251962, -0.12980273945998086, -0.07305887991797164, -0.01607733323649853, 0.04095651886426608, 0.09785712449558184, 0.15443936526798585, 0.21051915854885608, 0.26591405635016885, 0.32044383889833666, 0.37393110095467896, 0.4262018289792971, 0.47708596726047064, 0.5264179711678413, 0.5740373457293075, 0.6197891677797014, 0.6635245899821685, 0.7051013250828134, 0.744384108822945, 0.7812451400029145, 0.8155644962660226, 0.8472305242495614, 0.8761402028338829, 0.9021994783076073, 0.925323570358609, 0.9454372478952101, 0.9624750738003586, 0.9763816178223569, 0.9871116369096883, 0.9946302224031144, 0.9989129136062691, 0.9999457773652011, 0.9977254533980041, 0.9922591652270358, 0.9835646966781659, 0.9716703340235195]

fig = plt.figure()
ax = fig.add_subplot(111)

# some X and Y data
x = np.arange(10000)
y = np.random.randn(10000)

li, = ax.plot(x, y)

# draw and show it
ax.relim() 
ax.autoscale_view(True,True,True)
fig.canvas.draw()
plt.show(block=False)

# loop to update the data
while True:
    try:
        y[:-10] = y[10:]
        y[-10:] = np.random.randn(10)

        # set the new data
        li.set_ydata(y)

        fig.canvas.draw()

        time.sleep(0.01)
    except KeyboardInterrupt:
        break