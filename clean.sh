for i in $(seq 1 10);
do
    yes "yes" | rm -r setup-11/activations-$i/weights*
done

for i in $(seq 1 10);
do
    yes "yes" | rm -r setup-12/activations-$i/weights*
done

yes "yes" | rm -r setup-12/activations-28
yes "yes" | rm -r setup-12/activations-29
yes "yes" | rm -r setup-12/activations-39
yes "yes" | rm -r setup-12/activations-40
yes "yes" | rm -r setup-12/activations-41
yes "yes" | rm -r setup-12/activations-42

for i in $(seq 1 10);
do
    yes "yes" | rm -r setup-13/activations-$i/weights*
done

for i in $(seq 1 40);
do
    yes "yes" | rm -r setup-14/activations-$i/weights*
done

yes "yes" | rm -r setup-101/*/activation*
yes "yes" | rm -r setup-101/*/weight*
yes "yes" | rm -r setup-102/*/activation*
yes "yes" | rm -r setup-102/*/weight*
yes "yes" | rm -r setup-103/*/activation*
yes "yes" | rm -r setup-103/*/weight*

yes "yes" | rm -r setup-111/*/activation*
yes "yes" | rm -r setup-111/*/weight*
yes "yes" | rm -r setup-112/*/activation*
yes "yes" | rm -r setup-112/*/weight*
yes "yes" | rm -r setup-113/*/activation*
yes "yes" | rm -r setup-112/*/weight*

yes "yes" | rm -r setup-123/*/activation*
yes "yes" | rm -r setup-123/*/weight*
yes "yes" | rm -r setup-133/*/activation*
yes "yes" | rm -r setup-133/*/weight*

yes "yes" | rm -r setup-20/*/activation*
yes "yes" | rm -r setup-20/*/weight*
