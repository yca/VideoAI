#-------------------------------------------------
#
# Project created by QtCreator 2015-06-25T13:52:20
#
#-------------------------------------------------

QT       += core gui script scripttools network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = VideoAI
TEMPLATE = app

DEFINES += DEBUG

include(build_config.pri)

SOURCES += main.cpp\
        mainwindow.cpp \
    scripting/scriptedit.cpp \
    scripting/scripthighlighter.cpp \
    scripting/tabsettings.cpp \
    scripting/textedit.cpp \
    widgets/imagewidget.cpp \
    datasetmanager.cpp \
    vision/pyramids.cpp \
    opencv/opencv.cpp \
    windowmanager.cpp \
    scriptmanager.cpp \
    common.cpp \
    widgets/userscriptwidget.cpp \
    vlfeat/vlfeat.cpp \
    snippets.cpp \
    svm/liblinear.cpp \
    svm/linear.cpp \
    svm/tron.cpp \
    vision/pyramidsvl.cpp \
    imps/oxfordretrieval.cpp \
    imps/caltechbench.cpp \
    caffe/caffecnn.cpp \
    lmm/buffercloner.cpp \
    lmm/bowpipeline.cpp \
    lmm/cnnpipeline.cpp

HEADERS  += mainwindow.h \
    scripting/scriptedit.h \
    scripting/scripthighlighter.h \
    scripting/tabsettings.h \
    scripting/textedit.h \
    widgets/imagewidget.h \
    datasetmanager.h \
    vision/pyramids.h \
    opencv/opencv.h \
    windowmanager.h \
    scriptmanager.h \
    common.h \
    widgets/userscriptwidget.h \
    vlfeat/vlfeat.h \
    snippets.h \
    svm/liblinear.h \
    svm/linear.h \
    svm/tron.h \
    vision/pyramidsvl.h \
    imps/oxfordretrieval.h \
    imps/caltechbench.h \
    caffe/caffecnn.h \
    lmm/buffercloner.h \
    lmm/bowpipeline.h \
    lmm/cnnpipeline.h \
    lmm/lmmelements.h

FORMS    += mainwindow.ui \
    widgets/userscriptwidget.ui

RESOURCES += \
    scripting/images.qrc

CONFIG += opencv2 vlfeat lmm caffe cuda

opencv2 {
    INCLUDEPATH += /usr/include/opencv
    LIBS += -lopencv_core \
        -lopencv_imgproc \
        -lopencv_highgui \
        -lopencv_ml \
        -lopencv_video \
        -lopencv_features2d \
        -lopencv_calib3d \
        -lopencv_objdetect \
        -lopencv_contrib \
        -lopencv_legacy \
        -lopencv_flann \
        -lopencv_nonfree
}

opencv3 {
    INCLUDEPATH += /home/amenmd/myfs/source-codes/oss/build_x86/usr/local/include/
    LIBS += -L/home/amenmd/myfs/source-codes/oss/build_x86/usr/local/lib/
    LIBS += -lopencv_core \
        -lopencv_imgproc \
        -lopencv_imgcodecs \
        -lopencv_highgui \
        -lopencv_ml \
        -lopencv_video \
        -lopencv_features2d \
        -lopencv_calib3d \
        -lopencv_objdetect \
        -lopencv_flann \
        -lopencv_cudaarithm \
        -lopencv_xfeatures2d \
}

openmp {
    QMAKE_CXXFLAGS += -fopenmp
    LIBS += -fopenmp
}

vlfeat {
    DEFINES += HAVE_VLFEAT
    INCLUDEPATH += $$VLFEAT_PATH
    LIBS += -L$$VLFEAT_PATH/bin/glnxa64 -lvl
}

lmm {
    include($$INSTALL_PREFIX/usr/local/include/lmm/lmm.pri)
    DEFINES += HAVE_LMM
    SOURCES += lmm/classificationpipeline.cpp opencv/cvbuffer.cpp
    HEADERS += lmm/classificationpipeline.h opencv/cvbuffer.h
} else {
    SOURCES += debug.cpp
    HEADERS += debug.h
}

caffe {
    INCLUDEPATH += $$CAFFE_PATH/include/
    LIBS += -L$$CAFFE_PATH/lib -lcaffe
    LIBS += -lglog -lgflags -lprotobuf
    LIBS += -lboost_system -lboost_thread -lhdf5 -lhdf5_cpp -lhdf5_hl
    LIBS += -L/usr/lib64/atlas -llmdb
    #-lsatlas
    DEFINES += HAVE_CAFFE
}

cuda {
    INCLUDEPATH += /usr/local/cuda/include
    LIBS += -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn
}

LIBS += -L/usr/lib/libblas -lblas

OTHER_FILES += \
    docs/pipelines.txt
