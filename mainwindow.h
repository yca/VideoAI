#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindowPriv;

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	explicit MainWindow(QWidget *parent = 0);
	~MainWindow();

private slots:
	void on_pushEvaluate_clicked();
	void scriptTextChanged(const QString &text);

protected:
	void addScriptObject(QObject *obj, const QString &name);

private:
	Ui::MainWindow *ui;
	MainWindowPriv *p;
	QHash<QString, QObject *> scriptObjects;
};

#endif // MAINWINDOW_H
