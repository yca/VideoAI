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
    debug.cpp \
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
    imps/caltechbench.cpp

HEADERS  += mainwindow.h \
    scripting/scriptedit.h \
    scripting/scripthighlighter.h \
    scripting/tabsettings.h \
    scripting/textedit.h \
    widgets/imagewidget.h \
    datasetmanager.h \
    vision/pyramids.h \
    opencv/opencv.h \
    debug.h \
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
    imps/caltechbench.h

FORMS    += mainwindow.ui \
    widgets/userscriptwidget.ui

RESOURCES += \
    scripting/images.qrc

CONFIG += opencv2 openmp vlfeat

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

LIBS += -L/usr/lib/libblas -lblas
