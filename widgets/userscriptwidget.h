#ifndef USERSCRIPTWIDGET_H
#define USERSCRIPTWIDGET_H

#include <QWidget>

class QListWidgetItem;

namespace Ui {
class UserScriptWidget;
}

class UserScriptWidget : public QWidget
{
	Q_OBJECT

public:
	explicit UserScriptWidget(QWidget *parent = 0);
	~UserScriptWidget();
	QString runText;

private slots:
	void on_pushSave_clicked();

	void on_pushRun_clicked();

	void on_listScripts_currentRowChanged(int currentRow);

	void on_listScripts_itemDoubleClicked(QListWidgetItem *item);

private:
	Ui::UserScriptWidget *ui;
};

#endif // USERSCRIPTWIDGET_H
