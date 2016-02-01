#ifndef CNNVISUALIZER_H
#define CNNVISUALIZER_H

#include <QWidget>

namespace Ui {
class CNNVisualizer;
}

class CaffeCnn;

class CNNVisualizer : public QWidget
{
	Q_OBJECT

public:
	explicit CNNVisualizer(QWidget *parent = 0);
	~CNNVisualizer();

private slots:
	void on_pushLoad_clicked();

	void on_pushShowLayers_clicked();

	void on_pushLoadImage_clicked();

private:
	Ui::CNNVisualizer *ui;
	QList<CaffeCnn *> models;
	CaffeCnn *cmodel;
};

#endif // CNNVISUALIZER_H
