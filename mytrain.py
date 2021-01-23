import gdrive.MyDrive.myloader as myl
import gdrive.MyDrive.mymodel as mymodel

class Training:
    def dotrain(self):
        for epoch in range(10):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(myl.trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs = inputs.to(mymodel.device)
                labels = labels.to(mymodel.device)
                    

                # zero the parameter gradients
                mymodel.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = mymodel.net(inputs)
                loss = mymodel.criterion(outputs, labels)
                loss.backward()
                mymodel.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            mymodel.scheduler.step()
        print('Finished Training')