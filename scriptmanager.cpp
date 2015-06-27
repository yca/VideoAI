#include "scriptmanager.h"
#include "windowmanager.h"
#include "datasetmanager.h"
#include "common.h"
#include "debug.h"

#include "opencv/opencv.h"

#include "widgets/imagewidget.h"

#include "scripting/scriptedit.h"

#include "vision/pyramids.h"

#include <QMetaMethod>
#include <QScriptEngine>

Q_DECLARE_METATYPE(Mat)

ScriptManager * ScriptManager::inst = NULL;

static void addQObject(QScriptEngine *e, QObject *obj, const QString &oname)
{
	QScriptValue iv = e->newQObject(obj);
	e->globalObject().setProperty(oname, iv);
	ffDebug() << oname << obj;
}

class ScriptManagerPriv
{
public:
	ScriptManagerPriv()
	{

	}

	DatasetManager *dm;
	Pyramids *pyr;
	WindowManager *wm;
	QScriptEngine *eng;
};

QScriptValue matToScriptValue(QScriptEngine *eng, const Mat &m)
{
	OpenCV *cvmat = new OpenCV(m);
	QScriptValue obj = eng->newQObject(cvmat, QScriptEngine::ScriptOwnership);
	///QScriptValue obj = eng->newObject();
	obj.setProperty("rows", m.rows);
	obj.setProperty("cols", m.cols);
	return obj;
}

void scriptToMat(const QScriptValue &obj, Mat &m)
{
	if (obj.isObject()) {
		OpenCV *cvmat = (OpenCV *)obj.toQObject();
		m = cvmat->getRefMat();
	}
	//m.rows = obj.property("rows").toInt32();
	//m.cols = obj.property("cols").toInt32();
}

ScriptManager::ScriptManager(QObject *parent) :
	QObject(parent),
	p(new ScriptManagerPriv)
{
	inst = this;
	p->dm = new DatasetManager(this);
	p->pyr = new Pyramids(this);
	p->wm = new WindowManager(this);

	p->eng = new QScriptEngine(this);

	addScriptObject(p->dm, "dm");
	addScriptObject(p->pyr, "pyr");
	addScriptObject(p->wm, "wm");
	addScriptObject(this, "sm");

	qScriptRegisterMetaType(p->eng, matToScriptValue, scriptToMat);
	addQObject(p->eng, new OpenCV(this), "cv");
	addQObject(p->eng, new Common(this), "cmn");
}

void ScriptManager::evaluateScript(const QString &text)
{
	p->eng->evaluate(text);
	if (p->eng->hasUncaughtException())
		ffDebug() << p->eng->uncaughtExceptionLineNumber() << p->eng->uncaughtException().toString();
}

void ScriptManager::setCurrentWindow(int id)
{
	p->wm->setCurrentImageWindow(id);
	addQObject(p->eng, p->wm->getCurrentImageWindow(), "iw");
}

void ScriptManager::setCurrentCell(int row, int col)
{
	p->wm->getCurrentImageWindow()->setCurrentCell(row, col);
}

void ScriptManager::addScriptObject(QObject *obj, const QString &name)
{
	scriptObjects.insert(name, obj);
	addQObject(p->eng, obj, name);
}
