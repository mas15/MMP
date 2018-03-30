# import csv
#
# with open('temp.csv', 'w', encoding='utf-8', newline='') as fw:
#     writer = csv.writer(fw)
#     with open('all_tweets.csv', 'r', encoding='utf8') as fr:
#         reader = csv.reader(fr, delimiter=",")
#         for line in reader:
#             id, text, date = line
#             if text.startswith("RT"):
#                 print(text)
#             else:
#                 writer.writerow(line)
# #
# with open('temp.csv', 'w', encoding='utf-8', newline='') as fw:
#     writer = csv.writer(fw)
#     with open('all_tweets.csv', 'r', encoding='utf8') as fr:
#         reader = csv.reader(fr, delimiter=",")
#         prev_id = False
#         prev_text = ""
#         for line in reader:
#             id, text, date = line
#
#             if text.endswith("...") and not text.startswith("RT"):
#                 while text[-1] == ".":
#                     text = text[:-1]
#
#                 print()
#                 print(text)
#                 print(prev_text)
#                 print()
#
#                 text += " " + prev_text
#                 prev_id, prev_text, prev_date = id, text, date
#             else:
#                 if prev_id:
#                     writer.writerow((prev_id, prev_text, prev_date))
#                 prev_id, prev_text, prev_date = id, text, date

# at = None
# at2 = None
# with open("attr_po_6_wr_nb_nc_noRT", "r") as f:
#     f.readline()
#     at = [attr for attr in f.readlines()]
# with open("attr_po_6_wr_nb_bf_nc", "r") as f:
#     f.readline()
#     at2 = [attr for attr in f.readlines()]
# for a in at2:
#     if a not in at:
#         print(a)
# print("WSPOLNE " + str(len([a for a in at if a in at2])))
# print("tylko nowy " + str(len([a for a in at2 if a not in at])))
# print("tylko nowy " + str([a for a in at2 if a not in at]))
# print("tylko stary " + str(len([a for a in at if a not in at2])))
# print("tylko stary " + str([a for a in at if a not in at2]))

result = [(30, -0.26930331927379436, 0.654669172932331), (31, -0.26205766876619974, 0.6506832611832614), (32, -0.25292379029913925, 0.6428320471798732), (33, -0.2430115201373378, 0.641241608632913), (34, -0.2251501863192476, 0.6285836627140974), (35, -0.24133376631186493, 0.6355246376811595), (36, -0.23982974910394284, 0.6430555555555557), (37, -0.25331346153846146, 0.6439384615384615), (38, -0.2440273918953162, 0.6478009768009766), (39, -0.23918668921542463, 0.6531061031348385), (40, -0.2512202735356369, 0.6561498510004257), (41, -0.24288623296963574, 0.6477305236270752), (42, -0.24869488536155204, 0.6486948853615521), (43, -0.2566257513221317, 0.6546520671116054), (44, -0.2627989573150862, 0.6563473444118604), (45, -0.26675127638189144, 0.6491964174477222), (46, -0.2581998463347343, 0.6477703984819736), (47, -0.2633122832056655, 0.6446936645870469), (48, -0.24811374842334621, 0.6311546840958608), (49, -0.23558554061495235, 0.6230499280793398), (50, -0.2413946464186359, 0.624076210664446), (51, -0.2505871467306219, 0.6286693385114438), (52, -0.22852573396139425, 0.6004469162274041), (53, -0.21938972209597873, 0.5962013162988773), (54, -0.22799629526984366, 0.5985426135596299), (55, -0.22899132699012753, 0.5966728258191674), (56, -0.22981137258266188, 0.5964006997288336), (57, -0.2281028444968733, 0.5936200858761836), (58, -0.22846022822493411, 0.5904511784511784), (59, -0.2330334477211211, 0.5924084477211211), (60, -0.22894882820824053, 0.5939930759958512), (61, -0.23049536202423498, 0.5929407768713966), (62, -0.23111648150511033, 0.5910302746085586), (63, -0.23541809087851923, 0.5921059889676912), (64, -0.23436182654363602, 0.5887922062904715), (65, -0.2330819336567297, 0.5866384190123782), (66, -0.2324307064299666, 0.5823271867612295), (67, -0.234361509795175, 0.5827221655328799), (68, -0.2337953035685819, 0.581356279178338), (69, -0.2362857674796826, 0.5816672935841003), (70, -0.2371443741374371, 0.5797738562091502), (71, -0.22970702235402318, 0.5735805401405845), (72, -0.23020659640465807, 0.5765862441541688), (73, -0.229775619968625, 0.5747368602787025), (74, -0.23061132959151515, 0.5747795896297561), (75, -0.2250841067398356, 0.5729101936963573), (76, -0.22187577153292742, 0.5708438765985935), (77, -0.2200838555893616, 0.5695262347715178), (78, -0.21699023374453286, 0.5675437392795882), (79, -0.21626088229936818, 0.5667195978956985), (80, -0.21622358505691847, 0.5671326759660094), (81, -0.2155207620867423, 0.5651584432461626), (82, -0.2129559032169394, 0.5613313544804773), (83, -0.20857494245016917, 0.5567892281644549), (84, -0.20874080703011078, 0.5550305596802875), (85, -0.20196852247083724, 0.5504799060084905), (86, -0.19454797720756833, 0.5406349337293075), (87, -0.18741293578678386, 0.5362903105708944), (88, -0.1897641131266498, 0.5356545240855539), (89, -0.18459715249391923, 0.5292491049557189), (90, -0.18512496727565925, 0.5291384580007857), (91, -0.18135270792657043, 0.525834313277741), (92, -0.17722266939281361, 0.5238230010677721), (93, -0.17143091131250426, 0.5190421139484186), (94, -0.16846721859218172, 0.5176475464610342), (95, -0.16386058479833876, 0.5134540807332981), (96, -0.16228884783695557, 0.5115711444876733), (97, -0.15974527969595614, 0.5104561801698898), (98, -0.15338972469899204, 0.5029195052632554), (99, -0.15124363166353827, 0.5011658711658711), (100, -0.15136768461450667, 0.5042182239057239), (101, -0.14766749891291192, 0.49934945609945625), (102, -0.15054109025076123, 0.5006171937515222), (103, -0.1521215559882536, 0.5015191463496994), (104, -0.15474610322854393, 0.5039998345718275), (105, -0.15405593580060117, 0.5036855654302308), (106, -0.14878197574359375, 0.5001728981447651), (107, -0.1482287343392441, 0.4980110275177637), (108, -0.15304914776121803, 0.5013150437149753), (109, -0.15441319887311533, 0.5011180699332872), (110, -0.15726726312289685, 0.5048456107012445), (111, -0.15781396003012665, 0.5072457782119448), (112, -0.15779692476132123, 0.5075853591759898), (113, -0.1544076622116679, 0.5059482784581665), (114, -0.15277211120905165, 0.5043127274555502), (115, -0.15090969931022052, 0.502374134456664), (116, -0.15157788985663695, 0.5061485269757505), (117, -0.1462465769930255, 0.5028995262385674), (118, -0.1456007666880797, 0.5025489955709135), (119, -0.14541847603623426, 0.5013048089185076), (120, -0.1439833193217709, 0.4992996046515152), (121, -0.13856414146913748, 0.49417911473116954), (122, -0.13493652924734395, 0.49093652924734393), (123, -0.13391608905544494, 0.48849776236221787), (124, -0.13082356649198762, 0.48477093491304024), (125, -0.12732321743725133, 0.4820352593220681), (126, -0.12485064586125949, 0.4798571478118447), (127, -0.1187824221810555, 0.47324555284082265), (128, -0.12164667945147628, 0.4742850320898289), (129, -0.11938434587206298, 0.471497021928401), (130, -0.12218672144031428, 0.4742275377668449), (131, -0.11772073460997506, 0.46961946878719024), (132, -0.11641835067094791, 0.4686196085325831), (133, -0.1143522652805899, 0.46692441082638664), (134, -0.10822905126760218, 0.46153741581192176), (135, -0.10993365561684593, 0.4619237053680897), (136, -0.10513840176052924, 0.4582982530616445), (137, -0.10023384900570015, 0.4541179427171675), (138, -0.10043651664532999, 0.45381074977416436), (139, -0.10066088125461797, 0.45309990564486186), (140, -0.10176772446166399, 0.45370947203447953), (141, -0.1033792785456612, 0.4552535228020094), (142, -0.09583069611187123, 0.4484179403958664), (143, -0.09333445664407569, 0.4466278698177284), (144, -0.09416982639432142, 0.4473917834348942), (145, -0.09090000475600779, 0.44321317201579424), (146, -0.08794304854877455, 0.44137094452986203), (147, -0.08939643119153257, 0.44240705692943105), (148, -0.08741984378884199, 0.44112137140341307), (149, -0.0880259668195641, 0.4408981825288255), (150, -0.08916563649075626, 0.4403908406914564), (151, -0.08762621834094636, 0.43838115446173614), (152, -0.09078258812727219, 0.4391101544479181), (153, -0.08758213670846965, 0.43620598991947884), (154, -0.0849421555456038, 0.4335135841170324), (155, -0.08263467071097891, 0.43229298506177616), (156, -0.08364699438062478, 0.43285334358697397), (157, -0.08265575114590867, 0.4324870222370223), (158, -0.0844299411145028, 0.4330332930698101), (159, -0.08084557112852536, 0.43117816536355863), (160, -0.08030880493923004, 0.42909467690390996), (161, -0.07960845582927967, 0.4304856488117358), (162, -0.079226829090083, 0.43107868094193486), (163, -0.07425264293695555, 0.42642655598043383), (164, -0.07342024370241063, 0.42591482070891823), (165, -0.07505799129951496, 0.4274286809546874), (166, -0.07338331582773705, 0.4260714878707478), (167, -0.07315969731533373, 0.4267921759478124), (168, -0.06902008383901898, 0.4236526716984439), (169, -0.06945388464387026, 0.4243954361847842), (170, -0.07308007585199944, 0.4282597798689128), (171, -0.06898832520130105, 0.4261311823441582), (172, -0.06706550865874128, 0.4237600693281973), (173, -0.07089618436610867, 0.42573489404352804), (174, -0.07144411218175806, 0.4265842990976459), (175, -0.07019807494472169, 0.4272672183915741), (176, -0.0709249183072514, 0.42894960966527607), (177, -0.0740986514649572, 0.43241487323087097), (178, -0.07844863395855434, 0.4359562948370017), (179, -0.08041129380369216, 0.43646419309158635), (180, -0.08043415237016643, 0.43793666999353), (181, -0.08140296017471588, 0.4388326790502179), (182, -0.08245584716039539, 0.43952706080131815), (183, -0.08232365104572942, 0.4412468813548022), (184, -0.08280191971388368, 0.4413676567656765), (185, -0.0803326470926285, 0.439536627192131), (186, -0.07850163785485242, 0.4407918649032236), (187, -0.08077296082338054, 0.4429776852328294), (188, -0.08074509412945319, 0.44259366836740793), (189, -0.08312537629960659, 0.44418212776731697), (190, -0.08170976059636004, 0.44295610918448075), (191, -0.08450949583330014, 0.4443543066965397), (192, -0.08409612345207279, 0.44456123973114253), (193, -0.08387376063078134, 0.44356517818140817), (194, -0.08360749876815077, 0.44418442184507384), (195, -0.0804726820594624, 0.4409712439003061), (196, -0.07958203817382098, 0.4422714321132149), (197, -0.0803016758252616, 0.440941977143529), (198, -0.08199412280761093, 0.4422192635205565), (199, -0.0832618586177844, 0.4424036496625605), (200, -0.08376509986454594, 0.44250115934409984), (201, -0.08414000401300398, 0.44280543496759156), (202, -0.0852424241277292, 0.44317600346352254), (203, -0.08708278625957883, 0.44402850843069197), (204, -0.08844110248522186, 0.44499472301684423), (205, -0.08978473556025823, 0.44594911912190205), (206, -0.09040471048090998, 0.4452727723553413), (207, -0.09248599169974309, 0.44755845546785905), (208, -0.08923748982859869, 0.4454142256266149), (209, -0.08975829195125518, 0.4449741192893847), (210, -0.09090551321376628, 0.4454265517097376), (211, -0.09069262064313294, 0.44515690635741867), (212, -0.09076054989455018, 0.44459299196228635), (213, -0.09020050607302188, 0.44455219346200947), (214, -0.09178330811030705, 0.44607914513422203), (215, -0.0894908931603482, 0.4437311758458606), (216, -0.08978229202753035, 0.44402257471304274), (217, -0.08770508357945367, 0.4409564016813869), (218, -0.08799799290618204, 0.44119694119715486), (219, -0.08541492742499762, 0.4376876546977249), (220, -0.08295108031729742, 0.43573853676329044), (221, -0.08283340964201957, 0.43674645312028043), (222, -0.08277586042505053, 0.43607447153616163), (223, -0.0804532971235955, 0.4339537292757131), (224, -0.07705310165430107, 0.4322255154474045), (225, -0.0762216458263138, 0.4304778797042158), (226, -0.07662068225113089, 0.4295118387137159), (227, -0.07700038602578096, 0.4298417770351109), (228, -0.0769612761495918, 0.43005187733502787), (229, -0.07757397911809688, 0.4297699250640428), (230, -0.07578350025501934, 0.4276822344322345), (231, -0.07453911393617269, 0.42639096578802455), (232, -0.07207828800591953, 0.42472312427795983), (233, -0.0718541944136804, 0.4244990306857207), (234, -0.07139450217999993, 0.42453257749380746), (235, -0.06933408706065547, 0.422422567861991), (236, -0.06712126802000312, 0.4204047260416158), (237, -0.06708110447216681, 0.42031493531793795), (238, -0.06507815046486154, 0.4185050703657699), (239, -0.06532254373579865, 0.41787608574568336), (240, -0.0636641221789011, 0.4167505419319875), (241, -0.061244742387664475, 0.41428250921853477), (242, -0.06181786972997755, 0.4142768861234202), (243, -0.061434674982165205, 0.41389369137560783), (244, -0.06162882345867404, 0.4140409248486986), (245, -0.05977465473189947, 0.412186756121924), (246, -0.05987237857808386, 0.41252543980257367), (247, -0.05829405317706143, 0.41094711440155124), (248, -0.05712766738465902, 0.40997319583994357), (249, -0.05600464487742113, 0.40908906046183674), (250, -0.05092615485305663, 0.4050086124116662), (251, -0.05113823418300606, 0.4049346800634584), (252, -0.055634735258895396, 0.4082451770259637), (253, -0.05737454317185364, 0.4098925287833644), (254, -0.057537679056022195, 0.4105257268647871), (255, -0.059121188180839335, 0.4125763112944215), (256, -0.06121278934196883, 0.41382768633087536), (257, -0.06113507496963072, 0.41291373109611296), (258, -0.06048233654581153, 0.41170569881889757), (259, -0.06154985554948711, 0.4124962277892347), (260, -0.06259361120297585, 0.4139661602225837), (261, -0.0632469114747925, 0.41461946049440035), (262, -0.06212028959384036, 0.41395766254145566), (263, -0.05924916066251462, 0.4105372168686036), (264, -0.061111947938840006, 0.41313687005722005), (265, -0.06180683014487043, 0.41479364162663923), (266, -0.06347387383953629, 0.41641505031012455), (267, -0.0634612999230253, 0.41862308728820247), (268, -0.06303516380260565, 0.41765054841799026), (269, -0.06250749152100116, 0.41630641707740945), (270, -0.06304448999691509, 0.41652574477885845), (271, -0.06369089923358262, 0.41685642173548954), (272, -0.06386008224882628, 0.4164876526981789), (273, -0.06236950452404927, 0.4152212155506652), (274, -0.06078698877460953, 0.4143529978793136), (275, -0.058406275925761586, 0.41295173047121614), (276, -0.057737450489069975, 0.41201451332025846), (277, -0.05740243322339039, 0.4116320404741457), (278, -0.05666786540006702, 0.4108502316397053), (279, -0.05457837055381093, 0.4076963269775525), (280, -0.05219734182870278, 0.4052708050970686), (281, -0.05676273429795925, 0.40926460136293313), (282, -0.05229731541157362, 0.40497588684014507), (283, -0.05257195960373062, 0.40498831648105404), (284, -0.05270633108828343, 0.4050802183286395), (285, -0.054365580119604906, 0.40669718633722146), (286, -0.05308140686041646, 0.40485131836484123), (287, -0.05324155669788777, 0.40497076935425275), (288, -0.052098970048854, 0.4043048524017952), (289, -0.05203279705345509, 0.4047146486331759), (290, -0.05046398016579079, 0.4033190314103003), (291, -0.05000625171139439, 0.40522142311912596), (292, -0.04750355334565981, 0.40339874985221), (293, -0.04805499693559784, 0.40436849475852965), (294, -0.049452663049775025, 0.4059744021802098), (295, -0.04773952662150832, 0.404934031321436), (296, -0.046992761889756485, 0.4041872665896842), (297, -0.0479115333910084, 0.4059806976561381), (298, -0.04738747767318929, 0.405607147450648), (299, -0.04634324549977337, 0.4047661845678737)]
import matplotlib.pyplot as plt
x, y, y2 = zip(*result)
y = [-i for i in y]
plt.plot(x, y, x, y2)
plt.show()